import os

import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import input_data
# from scipy.misc import imsave as ims


class Draw():
    def __init__(self):
        self.genloss_history = []
        self.latloss_history = []

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.img_size = 28
        self.attention_n = 5
        self.n_hidden = 256
        self.n_z = 10
        self.sequence_length = 15
        self.batch_size = 64
        self.share_parameters = False
        self.step = 500
        self.epochs = 2500

        self.point_size = 3
        self.gx = None
        self.attention_points = np.zeros((self.sequence_length, self.batch_size,  self.point_size))
        self.attention_point = np.zeros((self.batch_size,  self.point_size))

        self.images = tf.placeholder(tf.float32, [None, 784])
        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        self.cs = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        x = self.images
        self.attn_params = []
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.img_size**2)) if t == 0 else self.cs[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            # r = self.read_basic(x,x_hat,h_dec_prev)
            r = self.read_attention(x,x_hat,h_dec_prev)
            print r.get_shape()

            # encode it to guass distrib
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(
                enc_state, tf.concat([r, h_dec_prev], axis=1)
            )
            # sample from the distrib to get z
            z = self.sampleQ(self.mu[t],self.sigma[t])
            print z.get_shape()
            # retrieve the hidden layer of RNN
            h_dec, dec_state = self.decode_layer(dec_state, z)

            print h_dec.get_shape()

            # map from hidden layer -> image portion, and then write it.
            # self.cs[t] = c_prev + self.write_basic(h_dec)
            self.cs[t] = c_prev + self.write_attention(h_dec)
            h_dec_prev = h_dec
            self.share_parameters = True # from now on, share variables

        # the final timestep
        self.generated_images = tf.nn.sigmoid(self.cs[-1])

        self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images),1))

        kl_terms = [0]*self.sequence_length
        for t in xrange(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.sequence_length*0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        self.cost = self.generation_loss + self.latent_loss
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def train(self):
        step_save = self.step
        attention_frame = np.zeros((self.sequence_length, self.batch_size, self.img_size**2))

        for i in xrange(self.epochs):
            xtrain, _ = self.mnist.train.next_batch(self.batch_size)
            cs, attn_params, gen_loss, lat_loss, _ = self.sess.run(
                [self.cs, self.attn_params, self.generation_loss, self.latent_loss, self.train_op],
                feed_dict={self.images: xtrain}
            )

            print "iter %d genloss %f latloss %f" % (i, gen_loss, lat_loss)
            self.genloss_history.append(gen_loss)
            self.latloss_history.append(lat_loss)

            # if i % step_save == 0:
            if i == self.epochs - 1:
                cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)

                results = None
                for cs_iter in xrange(self.sequence_length):
                    results = cs[cs_iter]
                    results_square = np.reshape(results, [-1, 28, 28])
                    print results_square.shape
                    ims("results/"+str(i)+"-step-"+str(cs_iter)+".jpg",merge(results_square,[8,8]))
                    for j in xrange(64):
                        center_x = int(attn_params[cs_iter][0][j][0])
                        center_y = int(attn_params[cs_iter][1][j][0])
                        distance = int(attn_params[cs_iter][2][j][0])
                        self.attention_point[j] = [center_x, center_y, distance]

                # Add results to attention_frame
                    attention_frame[cs_iter] = results
                    self.attention_points[cs_iter] = self.attention_point

                np.save("attention_frame.npy", attention_frame)
                np.save("attention_points.npy", self.attention_points)

        # # transform tensors to numpy arrays
        # np.save("attention_frame.npy", attention_frame)
        # np.save("attention_points.npy", self.attention_points)
        np.save("genloss_history_mnist_attn.npy", np.array(self.genloss_history))
        np.save("latloss_history_mnist_attn.npy", np.array(self.latloss_history))

    # given a hidden decoder layer:
    # locate where to put attention filters
    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters,5,axis=1)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.img_size+1)/2 * (gx_ + 1)
        gy = (self.img_size+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        # returns [Fx, Fy, gamma]

        self.attn_params.append([gx, gy, delta])

        return self.filterbank(gx,gy,sigma2,delta) + (tf.exp(log_gamma),)

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical guassian
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32),[1, -1])
        # centers for the individual patches
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im = tf.reshape(tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        Fy = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),1e-8)
        return Fx, Fy


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], axis=1)

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0,2,1])
            img = tf.reshape(img, [-1, self.img_size, self.img_size])
            # Apply the gaussian patches:
            # keep in mind: horiz = imgsize = verts (they are all the image size)
            # keep in mind: attn = height/length of attention patches
            # allfilters = [attn, vert] * [imgsize,imgsize] * [horiz, attn]
            # we have batches, so the full batch_matmul equation looks like:
            # [1, 1, vert] * [batchsize,imgsize,imgsize] * [1, horiz, 1]
            glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
            glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])
            # finally scale this glimpse w/ the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat], axis=1)

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder",reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_z)
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state


    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.img_size**2)
        return decoded_image_portion

    def write_attention(self, hidden_layer):
        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n**2)
        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0,2,1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w, Fx))
        wr = tf.reshape(wr, [self.batch_size, self.img_size**2])
        return wr * tf.reshape(1.0/gamma, [-1, 1])

   # def view(self):
   #      saver = tf.train.Saver(max_to_keep=2)
   #      saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
   #
   #      cs, attn_params, gen_loss, lat_loss = self.sess.run([self.cs, self.attn_params, self.generation_loss, self.latent_loss], feed_dict={self.images: base})
   #      print "genloss %f latloss %f" % (gen_loss, lat_loss)
   #
   #      cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)
   #
   #      print np.shape(cs)
   #      print np.shape(attn_params)
   #          # cs[0][cent]
   #
   #      for cs_iter in xrange(10):
   #          results = cs[cs_iter]
   #          results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_colors])
   #
   #          print np.shape(results_square)
   #
   #          for i in xrange(64):
   #              center_x = int(attn_params[cs_iter][0][i][0])
   #              center_y = int(attn_params[cs_iter][1][i][0])
   #              distance = int(attn_params[cs_iter][2][i][0])
   #
   #              size = 2;
   #
   #              # for x in xrange(3):
   #              #     for y in xrange(3):
   #              #         nx = x - 1;
   #              #         ny = y - 1;
   #              #
   #              #         xpos = center_x + nx*distance
   #              #         ypos = center_y + ny*distance
   #              #
   #              #         xpos2 = min(max(0, xpos + size), 63)
   #              #         ypos2 = min(max(0, ypos + size), 63)
   #              #
   #              #         xpos = min(max(0, xpos), 63)
   #              #         ypos = min(max(0, ypos), 63)
   #              #
   #              #         results_square[i,xpos:xpos2,ypos:ypos2,0] = 0;
   #              #         results_square[i,xpos:xpos2,ypos:ypos2,1] = 1;
   #              #         results_square[i,xpos:xpos2,ypos:ypos2,2] = 0;
   #              # print "%f , %f" % (center_x, center_y)
   #
   #          print results_square
   #
   #          ims("results/view-clean-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))



if __name__ == "__main__":
    model = Draw()
    model.train()
