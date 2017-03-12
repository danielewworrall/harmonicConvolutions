import tensorflow as tf
import math

class AffineVolumeTransformer(object):
    """Spatial Affine Volume Transformer Layer
    Implements a spatial transformer layer for volumetric 3D input.
    Implemented by Daniyar Turmukhambetov.
    """

    def __init__(self, out_size, name='SpatialAffineVolumeTransformer', interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of three ints
            The size of the output of the spatial network (depth, height, width), i.e. z, y, x
        name : string
            The scope name of the variables in this network.

        """
        self.name = name
        self.out_size = out_size
        self.param_dim = 3*4
        self.interp_method=interp_method

        with tf.variable_scope(self.name):
            self.voxel_grid = _meshgrid3d(self.out_size)
        
    
    def transform(self, inp, theta):
        """
        Affine Transformation of input tensor inp with parameters theta

        Parameters
        ----------
        inp : float
            The input tensor should have the shape 
            [batch_size, depth, height, width, in_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 12].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.]])
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        with tf.variable_scope(self.name):
            x_s, y_s, z_s = self._transform(inp, theta)
    
            output = _interpolate3d(
                inp, x_s, y_s, z_s,
                self.out_size,
                method=self.interp_method
                )
    
            batch_size, _, _, _, num_channels = inp.get_shape().as_list()
            output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], self.out_size[2], num_channels])

        return output


    
    def _transform(self, inp, theta):
        with tf.variable_scope(self.name + '_affine_volume_transform'):
            batch_size, _, _, _, num_channels = inp.get_shape().as_list()

            theta = tf.reshape(theta, (-1, 3, 4))
            voxel_grid = tf.tile(self.voxel_grid, [batch_size])
            voxel_grid = tf.reshape(voxel_grid, [batch_size, 4, -1])
    
            # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
            T_g = tf.matmul(theta, voxel_grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            z_s_flat = tf.reshape(z_s, [-1])
            return x_s_flat, y_s_flat, z_s_flat
    

"""
Common Functions

"""

def _meshgrid3d(out_size):
    """
    the regular grid of coordinates to sample the values after the transformation
    
    """
    with tf.variable_scope('meshgrid3d'):

        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        #z_t, y_t, x_t = tf.meshgrid(tf.linspace(0., out_size[0]-1.,  out_size[0]),
        #                       tf.linspace(0., out_size[1]-1.,  out_size[1]), 
        #                       tf.linspace(0., out_size[2]-1.,  out_size[2]), indexing='ij')

        z_t, y_t, x_t = tf.meshgrid(tf.linspace(-1., 1.,  out_size[0]),
                               tf.linspace(-1., 1.,  out_size[1]), 
                               tf.linspace(-1., 1.,  out_size[2]), indexing='ij')

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        z_t_flat = tf.reshape(z_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, z_t_flat, ones], 0)

        # Tiling for batches
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        return grid



def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        #rep = tf.transpose(tf.expand_dims(tf.ones(shape=[n_repeats, ]), 1), [1, 0])
        #rep = tf.cast(rep, tf.int32)
        #x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        #return tf.reshape(x, [-1])
        rep = tf.tile(tf.expand_dims(x,1), [1, n_repeats])
        return tf.reshape(rep, [-1])

def _interpolate3d(vol, x, y, z, out_size, method='bilinear'):
    return bilinear_interp3d(vol, x, y, z, out_size)

def bilinear_interp3d(vol, x, y, z, out_size, edge_size=1):
    with tf.variable_scope('bilinear_interp3d'):
        batch_size, depth, height, width, channels = vol.get_shape().as_list()

        if edge_size>0:
            vol = tf.pad(vol, [[0,0], [edge_size,edge_size], [edge_size,edge_size], [edge_size,edge_size], [0,0]], mode='CONSTANT')

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        z = tf.cast(z, tf.float32)

        depth_f  = tf.cast(depth, tf.float32)
        height_f = tf.cast(height, tf.float32)
        width_f  = tf.cast(width, tf.float32)

        out_depth  = out_size[0]
        out_height = out_size[1]
        out_width  = out_size[2]

        # scale indices to [0, width/height/depth - 1]
        x = (x + 1.) / 2. * (width_f  -1.)
        y = (y + 1.) / 2. * (height_f -1.)
        z = (z + 1.) / 2. * (depth_f  -1.)

        # clip to to [0, width/height/depth - 1] +- edge_size
        x = tf.clip_by_value(x, -edge_size, width_f  -1. + edge_size)
        y = tf.clip_by_value(y, -edge_size, height_f -1. + edge_size)
        z = tf.clip_by_value(z, -edge_size, depth_f  -1. + edge_size)

        x += edge_size
        y += edge_size
        z += edge_size

        # do sampling
        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        z0_f = tf.floor(z)
        x1_f = x0_f + 1
        y1_f = y0_f + 1
        z1_f = z0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        z0 = tf.cast(z0_f, tf.int32)

        x1 = tf.cast(tf.minimum(x1_f, width_f  - 1. + 2*edge_size),  tf.int32)
        y1 = tf.cast(tf.minimum(y1_f, height_f - 1. + 2*edge_size), tf.int32)
        z1 = tf.cast(tf.minimum(z1_f, depth_f  - 1. + 2*edge_size), tf.int32)

        dim3 = (width + 2*edge_size)
        dim2 = (width + 2*edge_size)*(height + 2*edge_size)
        dim1 = (width + 2*edge_size)*(height + 2*edge_size)*(depth + 2*edge_size)

        base = _repeat(tf.range(batch_size)*dim1, out_depth*out_height*out_width)
        base_z0 = base + z0*dim2
        base_z1 = base + z1*dim2

        base_y00 = base_z0 + y0*dim3
        base_y01 = base_z0 + y1*dim3
        base_y10 = base_z1 + y0*dim3
        base_y11 = base_z1 + y1*dim3

        idx_000 = base_y00 + x0
        idx_001 = base_y00 + x1
        idx_010 = base_y01 + x0
        idx_011 = base_y01 + x1
        idx_100 = base_y10 + x0
        idx_101 = base_y10 + x1
        idx_110 = base_y11 + x0
        idx_111 = base_y11 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        vol_flat = tf.reshape(vol, [-1, channels])
        I000 = tf.gather(vol_flat, idx_000)
        I001 = tf.gather(vol_flat, idx_001)
        I010 = tf.gather(vol_flat, idx_010)
        I011 = tf.gather(vol_flat, idx_011)
        I100 = tf.gather(vol_flat, idx_100)
        I101 = tf.gather(vol_flat, idx_101)
        I110 = tf.gather(vol_flat, idx_110)
        I111 = tf.gather(vol_flat, idx_111)

        # and finally calculate interpolated values
        w000 = tf.expand_dims((z1_f-z)*(y1_f-y)*(x1_f-x),1)
        w001 = tf.expand_dims((z1_f-z)*(y1_f-y)*(x-x0_f),1)
        w010 = tf.expand_dims((z1_f-z)*(y-y0_f)*(x1_f-x),1)
        w011 = tf.expand_dims((z1_f-z)*(y-y0_f)*(x-x0_f),1)
        w100 = tf.expand_dims((z-z0_f)*(y1_f-y)*(x1_f-x),1)
        w101 = tf.expand_dims((z-z0_f)*(y1_f-y)*(x-x0_f),1)
        w110 = tf.expand_dims((z-z0_f)*(y-y0_f)*(x1_f-x),1)
        w111 = tf.expand_dims((z-z0_f)*(y-y0_f)*(x-x0_f),1)

        output = tf.add_n([
            w000*I000, 
            w001*I001, 
            w010*I010, 
            w011*I011, 
            w100*I100, 
            w101*I101, 
            w110*I110, 
            w111*I111])
        return output


