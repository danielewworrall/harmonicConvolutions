require 'nn'
require 'cunn'
autograd = require 'autograd'
t = require 'torch'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local model = nn.Sequential()

local function permutationMatrix(nQuarterTurns)
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
	return rotation
end

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

params = {
	linear1 = {
		t.randn(1, 2, 3, 3),
		t.randn(2)
	},
}
local function gConv(weights)
	local productShape = weights:size()[1] * weights:size()[2]
	W = weights:view(productShape, 9)
	local W1 = (W * permutationMatrix(1)):viewAs(weights)
	local W2 = (W * permutationMatrix(2)):viewAs(weights)
	local W3 = (W * permutationMatrix(3)):viewAs(weights)
	local W4 = (W * permutationMatrix(4)):viewAs(weights)
	local gW = torch.cat({W1, W2, W3, W4}, 1)
	return gW
end

local function groupConvolution(weights, bias)
	--Invoke the spatial convolution operation to work with the group-shifted filters
	local gW = gConv(weights)
end

--autoLinear = autograd.nn.AutoModule('AutoLinear')(Linear, params.linear2[1]:clone(), params.linear2[2]:clone())

model:add(nn.SpatialConvolution(3, 10, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(32,32,1,1):ceil())
model:add(nn.View(10))
model:get(1).weight:normal(0,1)

--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

return model
