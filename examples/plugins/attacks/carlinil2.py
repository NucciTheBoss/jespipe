import jespipe.plugin.save as save
import numpy as np
import tensorflow as tf
from jespipe.plugin.attack.attack import Attack
from jespipe.plugin.start import start
from tensorflow.keras.models import load_model
from tqdm import trange


class CarliniL2(Attack):
    """
    This is a modified version of the L_2 optimized attack of Carlini and Wagner (2016).
    It has been modified to fit time series regression problems.
    """
    def __init__(self, model: str, features: np.ndarray, parameters: dict) -> None:
        """
        Create a Carlini&Wagner L_2 attack instance.

        ### Parameters:
        :param model: System file path to trained regressor model.
        :param model_test_features: Test features to use for adversarial example generation.
        :param parameters: Parameter dictionary for the attack.

        ### Methods:
        - public
          - attack (abstract): Launch L_2 attack on the given time series data.
        - private
          - _generate: Internal method to perform the L_2 attack on the given time series data.
          - _generate_batch: Internal method to generate batched adversarial samples and return them in an array.
        """
        self.model = load_model(model)
        self.features = features
        self.min_change = parameters["change"]
        self.learning_rate = parameters["learning_rate"]
        self.max_iter = parameters["max_iter"]
        self.binary_search_steps = parameters["binary_search_steps"]
        self.batch_size = parameters["batch_size"]
        self.initial_const = parameters["initial_const"]
        self.sequence_length = parameters["sequence_length"]
        self.verbose = parameters["verbose"]

    def attack(self) -> np.ndarray:
        """
        Launch L_2 attack on the given time series data.

        ### Returns:
        :return: An array holding the adversarial examples.
        """
        return self._generate(self.features)

    def _generate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Internal method to perform the L_2 attack on the given time series data.

        ### Parameters:
        :param x: An array with the original inputs to be attacked.

        ### Returns:
        :return: An array holding the adversarial examples.
        """
        pred = self.model.predict(x)
        self.mean = pred.mean()
        
        # Generate adversarial examples
        x_adv = np.zeros(x.shape)
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        for i in trange(nb_batches, desc="C&W L_2", disable = not self.verbose):
            index = i * self.batch_size
            x_adv[index:index+self.batch_size] = (self._generate_batch(x[index:index+self.batch_size]))
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
        # Initialize constant for binary search:
        c_current = np.ones(x.shape[0]) * self.initial_const
        c_best = np.zeros(x.shape[0])
        
        # Initialize placeholders for best l2 distance and attack found so far
        best_l2dist = np.inf * np.ones(x.shape[0])
        best_x_adv = x.copy()
        
        pred = self.model.predict(x)
        
        # Initialize boolean to decide if advesarial examples should predict above or below original
        # Since the adv examples are normalized between [0,1], adv examples that predict values approaching 0 or 1 are difficult to generate, hence the bool
        mean = pred.mean()
        above = (mean <= self.mean)
        
        if above:
            if mean + self.min_change > 0.9:
                above = False
        else:
            if mean - self.min_change < 0.1:
                above = True
        
        for bss in range(self.binary_search_steps):
            
            # Initialize variable to optimize
            w = tf.Variable(np.zeros(x.shape), trainable=True, dtype=tf.float32)
            
            for i_iter in range(self.max_iter):
                
                # Calculate loss
                with tf.GradientTape() as tape:
                    tape.watch(w)
                    
                    # Generate adversarial examples using w
                    x_adv = (tf.tanh(w) + 1.0) / 2.0
                    pred_adv = self.model(x_adv)
                    
                    # Calculate distance using the l2 metric
                    square_diff = tf.square(tf.subtract(x, x_adv))
                    l2dist = tf.reduce_sum(tf.reduce_sum(square_diff, axis=2), keepdims=(True))
                    
                    # Loss depends if adv prediction is meant to be above or below the benign prediction
                    if above:
                        f_sum = tf.add(tf.add(pred, self.min_change), tf.negative(pred_adv))
                    else:
                        f_sum = tf.add(tf.add(tf.negative(pred), self.min_change), pred_adv)
                    c_loss = tf.multiply(c_current, tf.maximum(f_sum, tf.zeros(x_adv.shape[0])))
                    
                    # Add the two sums from the loss function
                    loss = tf.add(l2dist, c_loss)
                
                # Calculate loss gradient w.r.t our optimization variable w 
                gradients = tape.gradient(loss, w)
                
                # Update w
                w = tf.subtract(w, tf.multiply(self.learning_rate, gradients))
                
                # Calculate l2dist and generate new adversarial predictions
                x_adv = x_adv.numpy()
                pred_adv = self.model.predict(x_adv)
                l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
                
                # Update adversarial examples if new best is found
                for e in range(x.shape[0]): 
                    if above:
                        if pred_adv[e] >= pred[e] + self.min_change and l2dist[e] <= best_l2dist[e]:
                            best_x_adv[e] = x_adv[e]
                            best_l2dist[e] = l2dist[e]
                    else:
                        if pred_adv[e] <= pred[e] - self.min_change and l2dist[e] <= best_l2dist[e]:
                            best_x_adv[e] = x_adv[e]
                            best_l2dist[e] = l2dist[e]
            
            pred_adv = self.model.predict(x_adv)
            
            # Update constant c using modified binary search
            for e in range(x.shape[0]):
                if above:
                    if pred_adv[e] >= pred[e] + self.min_change:
                        c_best[e] = c_current[e]
                        c_current[e] /= 2
                    else:
                        if c_best[e] == 0:
                            c_current[e] *= 10
                        else:
                            c_current[e] = (c_current[e] + c_best[e]) / 2
                else:
                    if pred_adv[e] <= pred[e] - self.min_change:
                        c_best[e] = c_current[e]
                        c_current[e] /= 2
                    else:
                        if c_best[e] == 0:
                            c_current[e] *= 10
                        else:
                            c_current[e] = (c_current[e] + c_best[e]) / 2
        return best_x_adv


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from pipeline
    if stage == "attack":
        attack = CarliniL2(parameters["model_path"], parameters["model_test_features"], parameters["attack_params"])
        result = attack.attack()
        save.adver_example(parameters["save_path"], parameters["attack_params"]["change"], result)

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
