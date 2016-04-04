t = require 'torch'
require 'nn'
timer = torch.Timer()

local GroupConvolution, parent = torch.class('nn.GroupConvolution', 'nn.Module')

function GroupConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self)
	assert(nOutputPlane % 4 == 0, "invalid input: " .. nOutputPlane .. " is not a multiple of 4")
  dW = dW or 1
  dH = dH or 1

  self.nIn = nInputPlane
  self.nOut = nOutputPlane
  self.kW = kW
  self.kH = kH

  self.dW = dW
  self.dH = dH
  self.padW = padW or 0
  self.padH = padH or self.padW
	
	-- Objects for matrix rotations
	self.productShape = self.nIn * self.nOut / 4
	self.kernelShape = self.kH * self.kW

  self.weight = torch.Tensor(self.nOut/4, self.nIn, kH, kW)
  self.bias = torch.Tensor(self.nOut)
	self.gradWeight = torch.Tensor(self.nOut/4, self.nIn, kH, kW)
  self.gradBias = torch.Tensor(self.nOut)
	
	-- The rotated weight
	self.perm = self:permutationMatrices()
	self.gW = self:groupRotate(self.weight)
	self.gradGW = torch.Tensor(self.nOut, self.nIn, kH, kW)
	
  self:reset()
end

function GroupConvolution:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1/math.sqrt(self.kW*self.kH*self.nIn)
	end
	if nn.oldSeed then
		self.weight:apply(function()
			 return torch.uniform(-stdv, stdv)
		end)
		if self.bias then
			 self.bias:apply(function()
			 return torch.uniform(-stdv, stdv)
			 end)
		end
	else
		self.weight:uniform(-stdv, stdv)
		if self.bias then
			 self.bias:uniform(-stdv, stdv)
		end
	end
end

function GroupConvolution:parameters()
	weights = {self.gW, self.weight, self.bias}
	gradWeights = {self.gradGW, self.gradWeight, self.gradBias}
	return weights, gradWeights
end

local function backCompatibility(self)
	self.finput = self.finput or self.gW.new()
	self.fgradInput = self.fgradInput or self.weight.new()
	if self.padding then
		 self.padW = self.padding
		 self.padH = self.padding
		 self.padding = nil
	else
		 self.padW = self.padW or 0
		 self.padH = self.padH or 0
	end
	if self.weight:dim() == 2 then
		 self.weight = self.weight:view(self.nOut, self.nIn, self.kH, self.kW)
	end
	if self.gW:dim() == 2 then
		self.gW = self.gW:view(self.nOut, self.nIn, self.kH, self.kW)
	end
	if self.gradGW and self.gradGW:dim() == 2 then
		 self.gradGW = self.gradGW:view(self.nOut, self.nIn, self.kH, self.kW)
	end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
	self.gW = self.gW:view(self.nOut, self.nIn * self.kH * self.kW)
	if self.gradGW and self.gradGW:dim() > 0 then
		 self.gradGW = self.gradGW:view(self.nOut, self.nIn * self.kH * self.kW)
	end
end

local function unviewWeight(self)
	self.gW = self.gW:view(self.nOut, self.nIn, self.kH, self.kW)
	if self.gradGW and self.gradGW:dim() > 0 then
		 self.gradGW = self.gradGW:view(self.nOut, self.nIn, self.kH, self.kW)
	end
end

function GroupConvolution:updateOutput(input)
	backCompatibility(self)
	self.gW = self:groupRotate(self.weight)
	viewWeight(self)
	input = makeContiguous(self, input)
	input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.gW:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
	unviewWeight(self)
	return self.output
end

function GroupConvolution:updateGradInput(input, gradOutput)
	if self.gradInput then
		backCompatibility(self)
		viewWeight(self)
		input, gradOutput = makeContiguous(self, input, gradOutput)
		input.THNN.SpatialConvolutionMM_updateGradInput(
			 input:cdata(),
			 gradOutput:cdata(),
			 self.gradInput:cdata(),
			 self.gW:cdata(),
			 self.bias:cdata(),
			 self.finput:cdata(),
			 self.fgradInput:cdata(),
			 self.kW, self.kH,
			 self.dW, self.dH,
			 self.padW, self.padH
		)
		unviewWeight(self)
		return self.gradInput
	end
end

function GroupConvolution:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	backCompatibility(self)
	input, gradOutput = makeContiguous(self, input, gradOutput)
	viewWeight(self)
	input.THNN.SpatialConvolutionMM_accGradParameters(
		 input:cdata(),
		 gradOutput:cdata(),
		 self.gradGW:cdata(),
		 self.gradBias:cdata(),
		 self.finput:cdata(),
		 self.fgradInput:cdata(),
		 self.kW, self.kH,
		 self.dW, self.dH,
		 self.padW, self.padH,
		 scale
	)
	unviewWeight(self)
	self.gradWeight = self:groupWeightUpdate(self.gradGW):mul(0.02)
end

function GroupConvolution:permutationMatrix(nQuarterTurns)
	--Generate a 3x3 permutation matrix for the p4 matrix rotation
	local permutation = t.zeros(9,9)
	permutation[1][7] = 1
	permutation[2][4] = 1
	permutation[3][1] = 1
	permutation[4][8] = 1
	permutation[5][5] = 1
	permutation[6][2] = 1
	permutation[7][9] = 1
	permutation[8][6] = 1
	permutation[9][3] = 1
	local rotation = permutation:clone()
	for i=1,nQuarterTurns do
		rotation = rotation * permutation
	end
	return rotation:cuda()
end

function GroupConvolution:permutationMatrices()
	local perm1 = self:permutationMatrix(1)
	local perm2 = self:permutationMatrix(2)
	local perm3 = self:permutationMatrix(3)
	local perm4 = self:permutationMatrix(4)
	local perm = torch.cat({perm1, perm2, perm3, perm4})
	return perm
end
	
function GroupConvolution:groupRotate(weights)
	local W = weights:view(self.productShape, self.kernelShape):cuda()
	local gW = torch.CudaTensor(self.productShape, self.kernelShape * 4)
	gW = gW:mm(W, self.perm)
	local _gW = gW:view(torch.LongStorage{self.nOut / 4, self.nIn, 4, self.kH, self.kW}):transpose(2,3)
	gW:copy(_gW)
	return gW
end

function GroupConvolution:groupWeightUpdate(gradGroupWeights)
	local gradWeights = torch.CudaTensor(self.productShape, self.kernelShape)
	local reshapedWeights = torch.CudaTensor(self.productShape, self.kernelShape*4)
	reshapedWeights:copy(gradGroupWeights:view(torch.LongStorage{self.nOut / 4, 4, self.nIn, self.kH*self.kW}):transpose(3,2))
	gradWeights:mm(reshapedWeights, self.perm:t())
	return gradWeights
end

function GroupConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function GroupConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nIn, self.nOut, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end