import tensorflow as tf
import numpy as np

class SiameseBaseClass(tf.keras.Model):
    def __init__(self, 
                 encoder_list, 
                 projector_list=None, 
                 encoder_indices=None, 
                 projector_indices=None, 
                 encoder_train_flag_list=None,
                 projector_train_flag_list=None,
                 loss_fn=None,
                 tracked_metrics=None,
                 rep_loss_flag=False,
                 *args,
                 **kwargs
                ):
        super(SiameseBaseClass, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.encoder_list      = encoder_list
        self.encoder_indices   = encoder_indices
        self.loss_tracker     = tf.keras.metrics.Mean(name=f"loss")
        self.tracked_metrics = {key: tf.keras.metrics.Mean(name=key) for key in tracked_metrics}
        
        self.projector_list    = projector_list
        self.projector_indices = projector_indices
                
        if len(self.encoder_indices)!=len(self.projector_indices):
            self.projector_indices+=['skip']*(len(self.encoder_indices)-len(self.projector_indices))
            
        if self.encoder_indices is None:
            self.encoder_indices = range(len(encoder_list))
        if self.projector_indices is None:
            self.projector_indices = range(len(projector_list))
            
        if encoder_train_flag_list is None:
            self.encoder_train_flag_list = [True]*len(encoder_list)
        else:
            self.encoder_train_flag_list = encoder_train_flag_list
        
        if projector_train_flag_list is None:
            self.projector_train_flag_list = [True]*len(projector_list)
        else:
            self.projector_train_flag_list = projector_train_flag_list
        
        self.rep_loss_flag = rep_loss_flag

    @property
    def metrics(self):
        return [self.loss_tracker, *self.tracked_metrics.values()]

    def train_step(self, data):
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_list = []
            for i, ds in enumerate(list(data)):
                z = self.encoder_list[self.encoder_indices[i]](ds, training=self.encoder_train_flag_list[self.encoder_indices[i]])
                z_list.append(z)
            
            y_list = []
            if self.projector_list:
                for i, ds in enumerate(list(z_list)):
                    if self.projector_indices[i]=='skip':
                        y_list.append(ds)
                    else:
                        y = self.projector_list[self.projector_indices[i]](ds, training=self.projector_train_flag_list[self.encoder_indices[i]])
                        y_list.append(y)
            else:
                y_list = z_list
                
            if self.rep_loss_flag:
                loss, metrics = self.loss_fn(y_list, z_list)
            else:
                loss, metrics = self.loss_fn(y_list)
            
        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        for name, value in metrics.items():
            self.tracked_metrics[name].update_state(value)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        
        # Forward pass through the encoder and predictor.
        z_list = []
        for i, ds in enumerate(list(data)):
            z = self.encoder_list[self.encoder_indices[i]](ds, training=False)
            z_list.append(z)

        y_list = []
        if self.projector_list:
            for i, ds in enumerate(list(z_list)):
                if self.projector_indices[i]=='skip':
                    y_list.append(ds)
                else:
                    y = self.projector_list[self.projector_indices[i]](ds, training=False)
                    y_list.append(y)
        else:
            y_list = z_list

        if self.rep_loss_flag:
            loss, metrics = self.loss_fn(y_list, z_list)
        else:
            loss, metrics = self.loss_fn(y_list)

        self.loss_tracker.update_state(loss)
        for name, value in metrics.items():
            self.tracked_metrics[name].update_state(value)
        
        return {m.name: m.result() for m in self.metrics}


class VICReg(SiameseBaseClass):
    @tf.function
    def calc_inv_loss(self, z_list):
        inv_loss=0
        for i, z_a in enumerate(z_list):
            for j, z_b in enumerate(z_list):
                if i!=j:
                    inv_loss+=self.inv_fn(z_a, z_b)
        return inv_loss

    @tf.function
    def calc_split_inv_loss(self, z_list):
        inv_loss=0
        for i, z in enumerate(z_list[1:]):
            inv_loss+=self.inv_fn(z_list[0][:,self.M[i]:self.M[i+1]], z)
        return inv_loss

    @tf.function
    def calc_loss(self, z_list):
        if self.split_rep:
            inv_loss = self.calc_split_inv_loss(z_list)
        else:
            inv_loss = self.calc_inv_loss(z_list)

        #Variance Loss.
        stds = [tf.math.sqrt(tf.math.reduce_variance(z, axis=0)+self.epsilon) for z in z_list]
        var_loss = [tf.reduce_mean(tf.keras.activations.relu(self.gamma-std_z_a)) for std_z_a in stds]
        var_loss = tf.reduce_sum(var_loss)

        #Covariance Loss.
        cov_loss = 0
        for i, z in enumerate(z_list):
            z -= tf.reduce_mean(z, axis=0)
            cov_z = tf.matmul(tf.transpose(z),z)/tf.cast(tf.shape(z)[0], tf.float32)
            cov_z = tf.linalg.set_diag(cov_z, tf.zeros(cov_z.shape[0:-1]))
            cov_z = tf.reduce_sum(cov_z**2)/z.shape[-1]
            cov_loss += cov_z

        #Total Loss.
        loss = self.lambd*inv_loss+self.mu*var_loss+self.nu*cov_loss

        return loss, {'inv_loss': inv_loss,
                      'var_loss': var_loss, 
                      'cov_loss': cov_loss
                     }

    
    def __init__(self, 
                 encoder_list, 
                 projector_list=None, 
                 encoder_indices=None, 
                 projector_indices=None, 
                 encoder_train_flag_list=None,
                 projector_train_flag_list=None,
                 lambd=25,
                 mu=25,
                 nu=1,
                 epsilon=1e-5,
                 gamma=1,
                 split_rep=False,
                 inv_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                 M=[4096, 4096]
                ):
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.gamma = gamma
        self.split_rep = split_rep
        self.M = np.cumsum([0] + M)
        self.inv_fn = inv_fn
        
        super().__init__(encoder_list, 
                               projector_list=projector_list, 
                               encoder_indices=encoder_indices, 
                               projector_indices=projector_indices, 
                               encoder_train_flag_list=encoder_train_flag_list,
                               projector_train_flag_list=projector_train_flag_list,
                               loss_fn=self.calc_loss,
                               tracked_metrics=['inv_loss', 'var_loss', 'cov_loss']
                              )
        
class SimCLR(SiameseBaseClass):
    def loss_func(self, z_list):
        if self.split_rep:
            loss, metrics = self.calc_split_loss(z_list)
        else:
            loss, metrics = self.calc_loss(z_list)
        return loss, metrics
        
    def calc_loss(self, z_list):
        losses = 0
        metrics = {}
        for i, i_rep in enumerate(z_list):
            for j, j_rep in enumerate(z_list):
                if i<j:
                    loss_i_j, loss_j_i = self.contrastive_loss(i_rep, j_rep)
                    metrics[f'{i}_{j}_loss'], metrics[f'{j}_{i}_loss'] = loss_i_j, loss_j_i
                    losses += (loss_i_j+loss_j_i)/2
        return losses, metrics
    
    def calc_split_loss(self, z_list):
        losses = 0
        metrics = {}
        for i, rep in enumerate(z_list[1:]):
            loss_0_i, loss_i_0 = self.contrastive_loss(z_list[0][:,self.M[i]:self.M[i+1]], rep)
            metrics[f'0_{i+1}_loss'], metrics[f'{i+1}_0_loss'] = loss_0_i, loss_i_0
            losses += (loss_0_i+loss_i_0)/2
        return losses, metrics
                    

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
    #     self.contrastive_accuracy.update_state(contrastive_labels, similarities)
    #     self.contrastive_accuracy.update_state(
    #         contrastive_labels, tf.transpose(similarities)
    #     )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return loss_1_2, loss_2_1
    
    
    def __init__(self, 
                 encoder_list, 
                 projector_list=None, 
                 encoder_indices=None, 
                 projector_indices=None, 
                 encoder_train_flag_list=None,
                 projector_train_flag_list=None,
                 temperature=0.5,
                 split_rep=False,
                 M=[4096, 4096]
                ):
        self.temperature = temperature
        self.split_rep = split_rep
        self.M = np.cumsum([0] + M)
        if self.split_rep:
            self.tracked_metrics = [f'0_{i+1}_loss' for i in range(len(projector_indices)-1)] + [f'{i+1}_0_loss' for i in range(len(projector_indices)-1)]
        else:
            self.tracked_metrics = [f'{i}_{j}_loss' for i in range(len(projector_indices)) for j in range(len(projector_indices)) if i!=j]
        
        super().__init__(encoder_list, 
                               projector_list=projector_list,
                               encoder_indices=encoder_indices,
                               projector_indices=projector_indices, 
                               encoder_train_flag_list=encoder_train_flag_list,
                               projector_train_flag_list=projector_train_flag_list,
                               loss_fn=self.loss_func,
                               tracked_metrics=self.tracked_metrics
                              )
 
