import tensorflow as tf

def test_generator(config, generator):
    """Tests the generator model architecture"""
    # Test model output shape
    noise = tf.random.normal([1, config.latent_dim])
    generated = generator(noise)
    assert generated.shape == (1, config.img_size, config.img_size, 3), "Incorrect generator output shape"
    
    # Test output range (should be -1 to 1 due to tanh)
    assert tf.reduce_min(generated) >= -1 and tf.reduce_max(generated) <= 1, "Generator output values out of range [-1, 1]"
    
def test_discriminator(config, discriminator):
    """Tests the discriminator model architecture"""
    # Test model output shape
    fake_image = tf.random.normal([1, config.img_size, config.img_size, 3])
    prediction = discriminator(fake_image)
    assert prediction.shape == (1, 1), "Incorrect discriminator output shape"
    
    # Test output range (should be 0 to 1 due to sigmoid)
    assert tf.reduce_min(prediction) >= 0 and tf.reduce_max(prediction) <= 1, "Discriminator output values out of range [0, 1]"
    
def test_color_diversity_loss(config, loss_value):
    """Tests the color diversity loss calculation"""
    assert isinstance(loss_value, tf.Tensor), "Loss must be a tensorflow tensor"
    assert loss_value.shape == (), "Loss should be a scalar"
    
def test_train_step(config, gen_loss, disc_loss):
    """Tests the output of a single training step"""
    assert isinstance(gen_loss, tf.Tensor), "Generator loss must be a tensorflow tensor"
    assert isinstance(disc_loss, tf.Tensor), "Discriminator loss must be a tensorflow tensor"
    assert gen_loss.shape == (), "Generator loss should be a scalar"
    assert disc_loss.shape == (), "Discriminator loss should be a scalar"
