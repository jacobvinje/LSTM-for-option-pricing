from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import keras as KER
from sklearn.model_selection import train_test_split
from keras.activations import tanh, relu

"""
NEW LSTM code below
"""
def create_model(config):
  """Builds an LSTM model of minimum 2 layers sequentially from a given config dictionary"""
  model = Sequential()

  model.add(LSTM(
    units = config["units"],
    activation = relu,
    input_shape = (config["seq_length"], config["num_features"]),
    return_sequences = True
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))


  for i in range(config["layers"]-2):
    model.add(LSTM(
      units = config["units"],
      activation = relu,
      return_sequences = True
    ))
    model.add(BatchNormalization(
      momentum = config["bn_momentum"]
    ))

  model.add(LSTM(
    units = config["units"],
    activation = relu,
    return_sequences = False
  ))

  model.add(BatchNormalization(
    momentum = config["bn_momentum"]
  ))

  model.add(Dense(
    units = 2,
    activation = relu
  ))  

  model.compile(
    optimizer = Adam(
      learning_rate = config["learning_rate"],
      clipnorm=config["clip_norm"]
    ),
    loss = "mse",
    metrics = ["mae"]
  )

  return model




"""
OLD LSTM code below
"""


class old_LSTM():
  """
  Implementation of the LSTM network
  """

  def __init(self, n_timesteps, n_features, n_batch, n_epochs, lr, dropout_rate, n_units, n_layers, folder_name):
    #Adjustable input parameters
    self.n_timesteps = n_timesteps
    self.n_features = n_features

    #Adjustable model parameters
    self.lr = lr
    self.n_batch = n_batch
    self.n_epochs = n_epochs
    self.dropout_rate = dropout_rate
    self.n_units = n_units
    self.n_layers = n_layers

    #Model paramters not intended for random search of hyper-paramater optimization
    self.optimizer = Adam
    self.loss = "mse"
    self.output_activation = "relu"

    #Other variables
    self.folder_name = folder_name

  def create_model(self):
    """
    Create a new LSTM model
    """    
    lstm_model = Sequential()

    #Input layer
    lstm_model.add(LSTM(units=self.n_units, input_shape=(self.n_timesteps, self.n_features), return_sequences=True, dropout = self.dropout_rate, recurrent_dropout = self.dropout_rate))
    
    #Hidden layers
    for i in range(self.layers -2):
      lstm_model.add(LSTM(units=self.n_units, return_sequences=True, dropout = self.dropout_rate, recurrent_dropout = self.dropout_rate))
    
    #Final LSTM layer
    lstm_model.add(LSTM(units=self.n_units, return_sequences=False, dropout = self.dropout_rate, recurrent_dropout = self.dropout_rate))

    #Output layer for final prediction of bid-ask prices
    lstm_model.add(Dense(units = 2, activation = self.output_activation))

    lstm_model.compile(optimizer = self.optimizer(lr=self.lr), loss = self.loss)
    self.model = lstm_model
 
  def fit(self, x_train, y_train):
    """
    Train the model based on a set of features X and output Y
    """
    self.model.fit(x_train, y_train, epochs = self.n_epochs, batch_size = self.n_batch)

  def get_prediction(self, x_test):
    """
    Takes a set of input features x_test and returns a model prediction
    """
    return self.model(x_test).numpy()[0] #TODO: verify output format

  def change_lr(self, new_lr):
    """
    Change the lr for a decaying lr schedule
    """
    K.set_value(self.model.optimizer.learning_rate, new_lr)
    self.lr = new_lr

  def save(self):
    """
    Save the network to file
    """
    path = "./Saves/" + self.folder_name
    self.model.save(path)
    print(path + "saved succesfully")

  def load(self):
    """
    Load a model from file
    """
    path = "./Saves/" + self.folder_name
    try:
      self.model = KER.models.load_model(path)
    except:
        raise ValueError("Failed to load model with path:", path)
    print(path + "loaded succesfully")