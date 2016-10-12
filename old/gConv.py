"""old gConv stuff"""


####### Very olde #######

def get_steerable_filter_asdfghjkj(V):
    """Return a steerable filter of order n from the input V"""
    Vsh = V[0].get_shape().as_list()     # [tap_length,i,o]
    k = int(np.sqrt(1 + 8.*Vsh[0]) - 2)
    masks = tf.reshape(get_basis_matrices(k, order=order), [k*k,Vsh[0]])
    
    W = []
    Vi = tf.reshape(V, [Vsh[0],Vsh[1]*Vsh[2]])
    Wx = tf.matmul(masks, Vi, name='Wx')
    W.append(tf.reshape(Wx, [k,k,Vsh[1],Vsh[2]]))
    W.append(-tf.reverse(tf.transpose(W[-1], perm=[1,0,2,3]), [False,True,False,False]))
    return tf.concat(3, W)

def gConv_(X, filter_size, n_filters, name=''):
    """Create a group convolutional module"""
    # Create variables
    k = filter_size
    n_channels = int(X.get_shape()[3])
    print('N_channels: %i' % (n_channels,))
    print('N_filters: %i' % (n_filters,))
    Q = get_weights([k,k,1,k*k], W_init=Q_init(), name=name+'_Q')
    V = get_weights([k*k,n_channels*n_filters], name=name+'_V')         # [h*w,c*f]
    # Project input X to Q-space
    Xq = channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID")   # [m,c,b,h',w']
    # Project V to Q-space: each col of Q is a filter transformation
    Q_ = tf.transpose(tf.reshape(Q, [k*k,k*k]))
    Vq = tf.matmul(Q_, V)
    
    Vq = tf.reshape(Vq, [1,k*k,n_channels,n_filters])                   # [1,m,c,f]
    Vq = tf.transpose(Vq, perm=[1,2,0,3])                               # [m,c,1,f]
    # Get angle
    Xqsh = tf.shape(Xq)                                                 # [m,c,b,h',w']
    Xq = to_filter_patch_pairs(Xq, Xqsh)                                # [m,c,bh'w',1]
    Vq, Xq = mutual_tile(Vq, Xq)    # Do we need a sanity check on this?# [m,c,bh'w',f]
    dot, ext = dot_ext_transform(Xq,Vq)                                 # [d,bh'w',f] [d,bh'w',f]
    angle = get_angle(dot[0,:,:], ext[0,:,:])                           # [bh'w',f]
    angle = tf.zeros_like(angle)
    # Get response
    response = get_response(angle, k, dot, ext, n_harmonics=4)
    # Reshape to image-like shape
    angle = fp_to_image(angle, Xqsh)                                    # [b,h',w',f]
    response = fp_to_image(response, Xqsh)                              # [b,h',w',f]
    return angle, response, V

def orthogonalize(Q):
    """Orthogonalize square Q"""
    Q = tf.reshape(Q, [9,9])
    S, U, V = tf.svd(Q, compute_uv=True, full_matrices=True)
    return tf.reshape(tf.matmul(U,tf.transpose(V)), [3,3,1,9])

def get_response(angle, k, dot, ext, n_harmonics=4):
    """Return the rotation response for the Lie Group up to n harmonics"""
    # Get response
    Rcos, Rsin = get_rotation_as_vectors(angle, k, n_harmonics=n_harmonics) # [d,bh'w',f]
    cos_response = tf.reduce_sum(dot*Rcos, reduction_indices=[0])       # [bh'w',f]
    sin_response = tf.reduce_sum(ext*Rsin, reduction_indices=[0])       # [bh'w',f]
    return cos_response + sin_response                                  # [bh'w',f]

def get_rotation_as_vectors(phi,k,n_harmonics=4):
    """Return the Jordan block rotation matrix for the Lie Group"""
    Rcos = []
    Rsin = []
    j = 1.
    for i in xrange(np.floor((k*k)/2.).astype(int)):
        if i >= n_harmonics:
            j = 0.
        Rcos.append(j*tf.cos((i+1)*phi))
        Rsin.append(j*tf.sin((i+1)*phi))
    if k % 2 == 1:
        Rcos.append(tf.ones_like(Rcos[-1]))
        Rsin.append(tf.zeros_like(Rsin[-1]))
    return tf.pack(Rcos), tf.pack(Rsin)

def channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID", name='conv'):
    """Convolve X with Q on each channel independently. Returns: tensor of
    shape [b,h',w',m*c].
    """
    Qsh = Q.get_shape().as_list()
    Xsh = X.get_shape().as_list()
    tile_shape = tf.pack([1,1,Xsh[3],1])
    Q = tf.tile(Q, tile_shape, name='Q_tile')
    Y = tf.nn.depthwise_conv2d(X, Q, strides=strides, padding=padding,
                               name=name+'chan_conv')
    return Y    
    
def channelwise_conv2d_(X, Q, strides=(1,1,1,1), padding="VALID"):
    """Convolve X with Q on each channel independently.
    
    X: input tensor of shape [b,h,w,c]
    Q: orthogonal tensor of shape [hw,hw]. Note h = w, m = hw
    
    returns: tensor of shape [m,c,b,h',w'].
    """
    Xsh = tf.shape(X)                                           # [b,h,w,c]
    X = tf.transpose(X, perm=[0,3,1,2])                         # [b,c,h,w]
    X = tf.reshape(X, tf.pack([Xsh[0]*Xsh[3],Xsh[1],Xsh[2],1])) # [bc,h,w,1]
    Z = tf.nn.conv2d(X, Q, strides=strides, padding=padding)    # [bc,h',w',m]
    Zsh = tf.shape(Z)
    Z = tf.reshape(Z, tf.pack([Xsh[0],Xsh[3],Zsh[1],Zsh[2],Zsh[3]])) # [b,c,h',w',m]
    return tf.transpose(Z, perm=[4,1,0,2,3])                    # [m,c,b,h',w']

def channelwise_conv2d_(X, Q, strides=(1,1,1,1), padding="VALID"):
    """Convolve X with Q on each channel independently. Using depthwise conv
    
    X: input tensor of shape [b,h,w,c]
    Q: orthogonal tensor of shape [hw,hw]. Note h = w, m = hw
    
    returns: tensor of shape [m,c,b,h',w'].
    """
    Xsh = tf.shape(X)
    Xsh_ = X.get_shape().as_list()
    Q_ = tf.tile(Q, [1,1,Xsh_[3],1])                             # [k,k,c,m]
    Z = tf.nn.depthwise_conv2d(X, Q_, strides=strides, padding=padding) # [b,h',w',c*k*k]
    Zsh = tf.shape(Z)
    Z_ = tf.reshape(Z, tf.pack([Xsh[0],Zsh[1],Zsh[2],Xsh[3],Zsh[3]/Xsh_[3]])) # [b,h',w',c,m]
    return tf.transpose(Z_, perm=[4,3,0,1,2])                    # [m,c,b,h',w']

def to_filter_patch_pairs(X, Xsh):
    """Convert tensor [m,c,b,h,w] -> [m,c,bhw,1]"""
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2]*Xsh[3]*Xsh[4],1]))

def from_filter_patch_pairs(X, Xsh):
    """Convert from filter-patch pairings"""
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2],Xsh[3],Xsh[4]]))

def fp_to_image(X, Xsh):
    """Convert from angular filter-patch pairings to standard image format"""
    return tf.reshape(X, tf.pack([Xsh[2],Xsh[3],Xsh[4],-1]))

def cart_to_polar(X, Y):
    """Input shape [m,:,:,:,:], output (r, theta). Assume d=9"""
    R = tf.sqrt(tf.pow(X,2.) + tf.pow(Y,2.))
    T = atan2(Y, X)
    return (R, T)

def polar_to_cart(R,T):
    """Polar to cartesian coordinates"""
    X = R*tf.cos(T)
    Y = R*tf.sin(T)
    return (X, Y)

def atan2(y, x, reg=1e-6):
    """Compute the classic atan2 function between y and x"""
    x = safe_reg(x)
    y = safe_reg(y)
    
    arg1 = y / (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) + x)
    z1 = 2*tf.atan(arg1)
    
    arg2 = (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) - x) / y
    z2 = 2*tf.atan(arg2)
    
    return tf.select(x>0,z1,z2)

def safe_reg(x, reg=1e-6):
    """Return the x, such that |x| >= reg"""
    return (2.*tf.to_float(tf.greater(x,0.))-1.)*(tf.abs(x) + reg)

def get_angle(dot, ext):
    """Get the angle in [0,2*pi] from one vector to another"""
    # Compute angles
    return modulus(atan2(ext, dot), 2*np.pi)

def dot_ext_transform(U,V):
    """Convert {U,V} to the dot-ext domain (vector representation of SO(N))"""
    # Dot: input [m,c,bh'w',f], [m,c,bh'w',f]
    dot = tf.reduce_sum(U*V,reduction_indices=[1])  # [m,bh'w',f]
    dotsh = tf.to_int32(tf.shape(dot)[0])
    seg_indices = tf.range(dotsh)/2
    dot = tf.segment_sum(dot, seg_indices)          # [ceil(m/2),bh'w',f]
    # Ext
    Vsh = tf.shape(V)
    V = tf.reshape(V, [Vsh[0],Vsh[1]*Vsh[2]*Vsh[3]])# [m,cbh'w'f]   
    V = tf.reshape(tf.matmul(blade_matrix(9),V), [Vsh[0],Vsh[1],Vsh[2],Vsh[3]]) # [m,c,bh'w',f]
    ext = tf.reduce_sum(U*V, reduction_indices=[1]) # [m,bh'w',f]
    return dot, tf.segment_sum(ext, seg_indices)    # [ceil(m/2),bh'w',f] [ceil(m/2),bh'w',f]

def blade_matrix(k):
    """Build the blade product matrix of order k"""
    blade = np.zeros([k,k])
    blade[k-1,k-1] = 1
    for i in xrange(int(np.floor(k/2.))):
        blade[(2*i)+1,2*i] = 1
        blade[2*i,(2*i)+1] = -1
    return tf.to_float(tf.identity(blade))
        
def mutual_tile(u,v):
    """Tile u and v to be the same shape"""
    ush = tf.shape(u)
    vsh = tf.shape(v)
    maxsh = tf.maximum(ush,vsh)
    u = tf.tile(u, maxsh/ush)
    v = tf.tile(v, maxsh/vsh)
    return u, v

def modulus(x,y):
    """Perform x % y and maintain sgn(x) = sgn(y)"""
    return x - y*tf.floordiv(x, y)

####### My old stuff #######

def gaussian_derivative(x,y,direction):
    return -2*direction*np.exp(-(x**2 + y**2))

def steer_conv(X, V, b=None, strides=(1,1,1,1), padding='VALID', k=3, n=2,
               name='steerConv'):
    Q = get_basis(k=k,n=n)
    Z = channelwise_conv2d(X, Q, strides=strides, padding=padding, name=name)
    # 1d convolution to combine filters
    Y = tf.nn.conv2d(Z, V, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    if b is not None:
        Y = tf.nn.bias_add(Y, b)
    return Y

def equi_steer_conv(X, V, strides=(1,1,1,1), padding='VALID', k=3, n=2,
                    name='equisteerConv'):
    """Steerable convolution returning max and argmax"""
    Xsh = tf.shape(X)
    Q = get_basis_(k=k,n=n)

    Z = channelwise_conv2d(X, Q, strides=strides, padding=padding, name=name)
    V_dot, V_blade = dot_blade_filter(V) 
    Y_dot = tf.nn.conv2d(Z, V_dot, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    Y_blade = tf.nn.conv2d(Z, V_blade, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    return Y_dot, Y_blade

def get_basis_(k=3, n=2):
    """Return a tensor of steerable filter bases"""
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    x, y = np.meshgrid(lin, lin)
    gdx = gaussian_derivative(x, y, x)
    gdy = gaussian_derivative(x, y, y)
    G0 = np.reshape(gdx/np.sqrt(np.sum(gdx**2)), [k,k,1,1])
    G1 = np.reshape(gdy/np.sqrt(np.sum(gdy**2)), [k,k,1,1])
    return to_constant_variable(np.concatenate([G0,G1], axis=3))

def get_basis(k=3, n=2):
    """Return a learnable steerable basis"""
    tap_length = int(((k+1)*(k+3))/8)
    tap = get_weights([tap_length], name='tap')
    masks = get_basis_masks(k)
    
    new_masks = []
    for i in xrange(tap_length):
        new_masks.append(masks[i]*tap[i])
    
    Wx = tf.reshape(tf.add_n(new_masks, name='Wx'), [k,k,1,1])
    Wy = tf.reverse(tf.transpose(Wx, perm=[1,0,2,3]), [False,True,False,False])
    return tf.concat(3, [Wx, Wy])

def get_basis_masks(k):
    """Return tf cosine masks for custom tap learning (works with odd sizes)"""
    tap_length = int(((k+1)*(k+3))/8)
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    R = X**2 + Y**2
    unique = np.unique(R)
    
    masks = []
    for i in xrange(tap_length):
        mask = (R == unique[i])*X/np.maximum(R,1.)
        masks.append(to_constant_variable(mask))
    return masks
