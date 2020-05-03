import matplotlib.pyplot as plt

def plot_train_data_errors(images, predictions, labels, show_predictions = True):
    '''
       Function that plots images, and overlays model predictions with true labels
       The function takes a list of 25 images
    '''

    fig = plt.figure(figsize = (15,15))

    for i,image in enumerate(images):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(image, cmap = 'gray')

        #If show predictions, then overlay the scatter on top of the labels
        if show_predictions:
            ax.scatter(*predictions[i].reshape(15,2).T, color = 'r', label = 'predictions - arg1')
        ax.scatter(*labels[i].reshape(15,2).T, color = 'g', label = 'labels - arg2')
    plt.legend()
