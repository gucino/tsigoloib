

#import library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import PIL.Image
import os

#some change
##################################################################################
##################################################################################
##################################################################################
#Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img
  
def run_deep_dream_simple(img, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))


  result = deprocess(img)
  display.clear_output(wait=True)

  return result
##################################################################################
##################################################################################
##################################################################################

#get images in the original folder
path = os.getcwd()
path = os.path.join(path,'original_image')
file_ls = os.listdir(path)


for file_name in file_ls:  
    if file_name.split('.')[-1] == 'png':
        #my image
        image = np.array(PIL.Image.open(f'original_image/{file_name}'))[:,:,:3]
        
        #pre-trained model originally used in DeepDream
        base_model = tf.keras.applications.InceptionV3 (include_top=False, weights='imagenet')
        
        #layer ls
        layer_ls = []
        for i, layer in enumerate(base_model.layers):
           layer_ls.append(layer.name)
           
        #Maximize the activations of these layers
        names = np.random.choice(layer_ls,3)
        layers = [base_model.get_layer(name).output for name in names]
        
        #Create the feature extraction model
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        deepdream = DeepDream(dream_model)
        dream_img = run_deep_dream_simple(img=image, steps=100, step_size=0.01)
    
        #save image
        dream_img = np.array(dream_img)
        i = PIL.Image.fromarray(dream_img)
        
        layer_name = ''
        for each in names:
            layer_name += each
        new_file_name = f'DREAM_ME/{file_name.split(".")[0]}_{layer_name}.png'
        i.save(new_file_name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
