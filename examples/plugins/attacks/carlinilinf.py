import jespipe.plugin.save as save
import numpy as np
import tensorflow as tf
from jespipe.plugin.attack.attack import Attack
from jespipe.plugin.start import start
from tensorflow.keras.models import load_model
from tqdm import trange


class CarliniLinf(Attack):
    """
    This is a modified version of the L_inf optimized attack of Carlini and Wagner (2016).
    It has been modified to fit time series regression problems.
    """
    def __init__(self, model: str, features: np.ndarray, parameters: dict) -> None:
        """
        Create a Carlini&Wagner L_inf attack instance.

        ### Parameters:
        :param model: System file path to trained regressor model.
        :param model_test_features: Test features to use for adversarial example generation.
        :param parameters: Parameter dictionary for the attack.

        ### Methods:
        - public
          - attack (abstract): Launch L_inf attack on the given time series data.
        - private
          - _generate: Internal method to perform the L_inf attack on the given time series data.
          - _generate_batch: Internal method to generate batched adversarial samples and return them in an array.
        """
        self.model = load_model(model)
        self.features = features
        self.min_change = parameters["change"]
        self.learning_rate = parameters["learning_rate"]
        self.max_iter = parameters["max_iter"]
        self.batch_size = parameters["batch_size"]
        self.initial_const = parameters["initial_const"]
        self.largest_const = parameters["largest_const"]
        self.sequence_length = parameters["sequence_length"]
        self.decrease_factor = parameters["decrease_factor"]
        self.verbose = parameters["verbose"]

    def attack(self) -> np.ndarray:
        """
        Launch L_inf attack on the given time series data.

        ### Returns:
        :return: An array holding the adversarial examples.
        """
        return self._generate(self.features)

    def _generate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Internal method to perform the L_inf attack on the given time series data.

        ### Parameters:
        :param x: An array with the original inputs to be attacked.

        ### Returns:
        :return: An array holding the adversarial examples.
        """
        num_rows, num_cols = x.shape
        
        # Reformat test set to include sequence length
        x_sequence = []
        for i in range(len(x)-self.sequence_length):
            s = []
            for j in range(0, self.sequence_length):
                s.append(x[[(i + j)], :])
            x_sequence.append(s)
        x_sequence = np.array(x_sequence)
        x_sequence = x_sequence.reshape(x_sequence.shape[0],self.sequence_length, num_cols)
        
        # Generate adversarial examples
        x_adv = np.zeros(x_sequence.shape)
        nb_batches = int(np.ceil(x_sequence.shape[0] / float(self.batch_size)))
        for i in trange(nb_batches, desc="C&W L_inf", disable = not self.verbose):
            index = i * self.batch_size
            x_adv[index:index+self.batch_size] = (self._generate_batch(x_sequence[index:index+self.batch_size]))
        print(x_adv.shape)
        return x_adv
    
    def _generate_batch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Internal method to generate batched adversarial samples and return them in an array.

        ### Parameters:
        :param x: An array with the batched original inputs to be attacked.

        ### Returns:
        :return: An array holding the batched adversarial examples.
        """
        
        # Initialize placeholders for best l2 distance and attack found so far
        best_linf_dist = np.inf * np.ones(x.shape[0])
        best_x_adv = x.copy()
        
        pred = self.model.predict(x)
        
        # Initialize boolean to decide if advesarial examples should predict above or below original
        # Since the adv examples are normalized between [0,1], adv examples that predict values approaching 0 or 1 are difficult to generate, hence the bool
        mean = pred.mean()
        above = (mean <= .5)
        
        # Initialize variable to optimize
        w = tf.Variable(np.zeros(x.shape), trainable=True, dtype=tf.float32)
        best_w = w
        x_adv = (tf.tanh(w) + 1.0) / 2.0
        
        # Initialize variables to iterate over
        c_current = self.initial_const
        tau = np.max(np.abs(x_adv - x))

        while tau > 1./256:
            
            # Using "warm-start", meaning start gradient descent from the solution found on the previous iteration
            w = best_w

            found = False
            
            while c_current < self.largest_const and not found:
                
                for i_iter in range(self.max_iter):
    
                    # Calculate loss
                    with tf.GradientTape() as tape:
                        tape.watch(w)
                        
                        # Generate adversarial examples using w
                        x_adv = (tf.tanh(w) + 1.0) / 2.0
                        pred_adv = self.model(x_adv)
                        
                        # Calculate the first loss term using tau
                        tau_loss = tf.reduce_sum(tf.maximum(0.0, (tf.abs(x_adv - x) - tau)))
                        
                        # Loss depends if adv prediction is meant to be above or below the benign prediction
                        if above:
                            f_sum = tf.add(tf.add(pred, self.min_change), tf.negative(pred_adv))
                        else:
                            f_sum = tf.add(tf.add(tf.negative(pred), self.min_change), pred_adv)
                        c_loss = tf.multiply(c_current, tf.maximum(f_sum, tf.zeros(x_adv.shape[0])))
                        
                        # Add the two sums from the loss function
                        loss = tf.add(tau_loss, c_loss)
                    
                    # Calculate loss gradient w.r.t our optimization variable w 
                    gradients = tape.gradient(loss, w)
                    
                    # Update w
                    w = tf.subtract(w, tf.multiply(self.learning_rate, gradients))
                    
                    # Calculate linf_dist and generate new adversarial predictions
                    x_adv = x_adv.numpy()
                    pred_adv = self.model.predict(x_adv)
                    linf_dist = np.max(np.abs(x_adv - x).reshape(x.shape[0], -1), axis=1)
                    
                    # Update adversarial examples if new best is found
                    for e in range(x.shape[0]): 
                        if above:
                            if pred_adv[e] >= pred[e] + self.min_change and linf_dist[e] < best_linf_dist[e]:
                                best_x_adv[e] = x_adv[e]
                                best_linf_dist[e] = linf_dist[e]
                                best_w = w
                                found = True
                        else:
                            if pred_adv[e] <= pred[e] - self.min_change and linf_dist[e] < best_linf_dist[e]:
                                best_x_adv[e] = x_adv[e]
                                best_linf_dist[e] = linf_dist[e]
                                best_w = w
                                found = True
                
                    # Early abort if there are no gradient updates
                    if tf.reduce_sum(gradients) == 0:
                        break
                    
                # Update constant c if no example has been found yet
                if not found:
                    c_current *= 2
            
            tau *= self.decrease_factor
            
        return best_x_adv


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "attack":
        attack = CarliniLinf(parameters["model_path"], parameters["model_test_features"], parameters["attack_params"])
        result = attack.attack()
        save.adver_example(parameters["save_path"], parameters["attack_params"]["change"], result)

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
