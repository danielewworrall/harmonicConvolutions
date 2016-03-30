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

  self.weight = torch.Tensor(nOutputPlane/4, nInputPlane, kH, kW)
  self.bias = torch.Tensor(nOutputPlane)
  self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self.gradBias = torch.Tensor(nOutputPlane)
	
	self.perm1 = self:permutationMatrix(1)
	self.perm2 = self:permutationMatrix(2)
	self.perm3 = self:permutationMatrix(3)
	self.perm4 = self:permutationMatrix(4)

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

function GroupConv:updateOutput(input)
	self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
	print(1)
	local gW = self:groupRotate(self.weight)
	print(2)
	input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      gW:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
	print(3)
	return self.output
end

function GroupConv:updateGradInput(input, gradOutput)
end

function GroupConv:accGradParameters(input, gradOutput)
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
	local productShape = weights:size()[1] * weights:size()[2]
	W = weights:view(productShape, 9):cuda()
	local W1 = (W * self.perm1):viewAs(weights)
	local W2 = (W * self.perm2):viewAs(weights)
	local W3 = (W * self.perm3):viewAs(weights)
	local W4 = (W * self.perm4):viewAs(weights)
	local gW = torch.cat({W1, W2, W3, W4}, 1)
	return gW
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
