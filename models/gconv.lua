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

--model:add(nn.SpatialConvolution(3, 40, 3, 3, 1, 1, 1, 1))
model:add(nn.GroupConv(3, 96, 3, 3, 1, 1, 1, 1))
model:add(nn.GroupConv(96, 96, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.GroupConv(96, 192, 3, 3, 1, 1, 1, 1))
model:add(nn.GroupConv(192, 192, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(16, 16, 1, 1):ceil())
model:add(nn.View(192))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(192, 10))


return model
