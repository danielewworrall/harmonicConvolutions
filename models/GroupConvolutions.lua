t = require 'torch'
require 'nn'

local GroupConv, parent = torch.class('nn.GroupConv', 'nn.Module')

function GroupConv:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self)
	assert(nOutputPlane % 4 == 0, "invalid input: " .. nOutputPlane .. " is not a multiple of 4")
  dW = dW or 1
  dH = dH or 1

  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kW = kW
  self.kH = kH

  self.dW = dW
  self.dH = dH
  self.padW = padW or 0
  self.padH = padH or self.padW
	
	-- Objects for matrix rotations
	self.productShape = self.nInputPlane * self.nOutputPlane / 4
	self.kernelShape = self.kH * self.kW
	self.W1 = torch.CudaTensor(self.productShape, self.kernelShape)
	self.W2 = torch.CudaTensor(self.productShape, self.kernelShape)
	self.W3 = torch.CudaTensor(self.productShape, self.kernelShape)
	self.W4 = torch.CudaTensor(self.productShape, self.kernelShape)

  self.weight = torch.Tensor(nOutputPlane/4, nInputPlane, kH, kW)
  self.bias = torch.Tensor(nOutputPlane)
  self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self.gradBias = torch.Tensor(nOutputPlane)
	
	-- The rotated weight
	self.perm1 = self:permutationMatrix(1)
	self.perm2 = self:permutationMatrix(2)
	self.perm3 = self:permutationMatrix(3)
	self.perm4 = self:permutationMatrix(4)
	self.gW = self:groupRotate(self.weight)
	
  self:reset()
end

function GroupConv:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
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

function GroupConv:parameters()
	weights = {self.gW, self.bias}
	gradWeights = {self.gradWeight, self.gradBias}
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
		 self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
	if self.gW:dim() == 2 then
		self.gW = self.gW:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
	if self.gradWeight and self.gradWeight:dim() == 2 then
		 self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
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
	self.gW = self.gW:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
	if self.gradWeight and self.gradWeight:dim() > 0 then
		 self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
	end
end

local function unviewWeight(self)
	self.gW = self.gW:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	if self.gradWeight and self.gradWeight:dim() > 0 then
		 self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
end

function GroupConv:updateOutput(input)
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

function GroupConv:updateGradInput(input, gradOutput)
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

function GroupConv:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	backCompatibility(self)
	input, gradOutput = makeContiguous(self, input, gradOutput)
	viewWeight(self)
	input.THNN.SpatialConvolutionMM_accGradParameters(
		 input:cdata(),
		 gradOutput:cdata(),
		 self.gradWeight:cdata(),
		 self.gradBias:cdata(),
		 self.finput:cdata(),
		 self.fgradInput:cdata(),
		 self.kW, self.kH,
		 self.dW, self.dH,
		 self.padW, self.padH,
		 scale
	)
	unviewWeight(self)
	self:groupWeightUpdate(self.gradWeight)
end

function GroupConv:permutationMatrix(nQuarterTurns)
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

function GroupConv:groupRotate(weights)
	local W = weights:view(self.productShape, self.kernelShape):cuda()
	self.W1:mm(W, self.perm1)
	self.W2:mm(W, self.perm2)
	self.W3:mm(W, self.perm3)
	self.W4:mm(W, self.perm4)
	local gW = torch.cat({W1, W2, W3, W4}, 2)
	return gW
end

function GroupConv:groupWeightUpdate(gradGroupWeights)
	gradGroupWeights = torch.reshape(gradGroupWeights,
																	 torch.LongStorage{self.nOutputPlane/4, self.nInputPlane, 4, self.kH, self.kW})
	gW1 = torch.reshape(gradGroupWeights[{{},{},1,{},{}}], self.productShape, self.kernelShape) * self.perm1:t()
	gW2 = torch.reshape(gradGroupWeights[{{},{},2,{},{}}], self.productShape, self.kernelShape) * self.perm2:t()
	gW3 = torch.reshape(gradGroupWeights[{{},{},3,{},{}}], self.productShape, self.kernelShape) * self.perm3:t()
	gW4 = torch.reshape(gradGroupWeights[{{},{},4,{},{}}], self.productShape, self.kernelShape) * self.perm4:t()
	gradWeights = gW1 + gW2 + gW3 + gW4
	self.weight = self.weight - 0.2 * gradWeights --need to change to a gradParameter
end

function GroupConv:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function GroupConv:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

function GroupConv:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
