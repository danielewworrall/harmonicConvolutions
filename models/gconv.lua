require 'nn'
require 'cunn'
require './GroupConvolutions'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

--model:add(nn.SpatialConvolution(3, 10, 3, 3, 1, 1, 1, 1))
model:add(nn.GroupConv(3, 40, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(32,32,1,1):ceil())
model:add(nn.View(40))
model:add(nn.Linear(40, 10))

--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

return model
