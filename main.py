from sklearn import datasets
import random
import math
import time
import PySimpleGUI as sg


sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Select topology: ')],
            [sg.Radio('5 - 2 - 3', "RADIO1", default=True),
             sg.Radio('5 - 6 - 3', "RADIO1")],
            [sg.Button('Ok'), sg.Button('Cancel')] ]


# Create the Window
window = sg.Window('ANN', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break

window.close()

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


# Return the target and index based on the target array
def get_output(netout1, netout2, netout3):

    if netout1 > netout2 and netout1 > netout3:
        return 0

    elif netout2 > netout1 and netout2 > netout3:
        return 1

    elif netout3 > netout1 and netout3 > netout2:
        return 2


def target_output(target, index: int):
    if target == 0 and index == 0:
        return 1

    elif target == 1 and index == 1:
        return 1

    elif target == 2 and index == 2:
        return 1

    else:
        return 0


# Importing the iris data set
iris = datasets.load_iris()
biasNeuron = 1

if values[0]:
    # Bias neuron is always 1
    input1 = 0
    input2 = 0
    input3 = 0
    input4 = 0

    # Initialize the weights randomly

    h1weight0 = random.uniform(-0.5, 0.5)
    h1weight1 = random.uniform(-0.5, 0.5)
    h1weight2 = random.uniform(-0.5, 0.5)
    h1weight3 = random.uniform(-0.5, 0.5)
    h1weight4 = random.uniform(-0.5, 0.5)

    h2weight0 = random.uniform(-0.5, 0.5)
    h2weight1 = random.uniform(-0.5, 0.5)
    h2weight2 = random.uniform(-0.5, 0.5)
    h2weight3 = random.uniform(-0.5, 0.5)
    h2weight4 = random.uniform(-0.5, 0.5)

    h_output1weight1 = random.uniform(-0.5, 0.5)
    h_output1weight2 = random.uniform(-0.5, 0.5)
    h_output1weight3 = random.uniform(-0.5, 0.5)

    h_output2weight1 = random.uniform(-0.5, 0.5)
    h_output2weight2 = random.uniform(-0.5, 0.5)
    h_output2weight3 = random.uniform(-0.5, 0.5)

    h1inputSum = 0

    # Learning rate
    learning_rate = 0.3

    # Initial epoch
    epoch = 1

    selected_index = 0
    # Training data

    start1 = time.time()
    while selected_index < 150:
        input1 = iris.data[selected_index][0]
        input2 = iris.data[selected_index][1]
        input3 = iris.data[selected_index][2]
        input4 = iris.data[selected_index][3]


        h1inputSum = (biasNeuron * h1weight0) + (input1 * h1weight1) + (input2 * h1weight2) + \
                     (input3 * h1weight3) + (input4 * h1weight4)
        h2inputSum = (biasNeuron * h2weight0) + (input1 * h2weight1) + (input2 * h2weight2) + \
                     (input3 * h2weight3) + (input4 * h2weight4)

        # Sigmoid function to output from hidden layer
        h1output = sigmoid_function(h1inputSum)
        h2output = sigmoid_function(h2inputSum)


        # Try without bias for now (biasNeuron * h1weight0) +
        output1WeightedSum = (h1output * h_output1weight1) + (h2output * h_output2weight1)
        output2WeightedSum = (h1output * h_output1weight2) + (h2output * h_output2weight2)
        output3WeightedSum = (h1output * h_output1weight3) + (h2output * h_output2weight3)

        output1 = sigmoid_function(output1WeightedSum)
        output2 = sigmoid_function(output2WeightedSum)
        output3 = sigmoid_function(output3WeightedSum)

        target1 = target_output(iris.target[selected_index], 0)
        target2 = target_output(iris.target[selected_index], 1)
        target3 = target_output(iris.target[selected_index], 2)

        error_output1 = output1 * (1 - output1) * (target1 - output1)
        error_output2 = output2 * (1 - output2) * (target2 - output2)
        error_output3 = output3 * (1 - output3) * (target3 - output3)

        error_hidden1 = (h_output1weight1 * error_output1) + (h_output1weight2 * error_output2) + (h_output1weight3 * error_output3)
        error_hidden2 = (h_output2weight1 * error_output1) + (h_output2weight2 * error_output2) + (h_output2weight3 * error_output3)

        error_hidden1result = error_hidden1 * (h1output * (1 - h1output))
        error_hidden2result = error_hidden2 * (h2output * (1 - h2output))

        # Updating weights through backpropagation
        h1weight0 += learning_rate * error_hidden1result * biasNeuron
        h1weight1 += learning_rate * error_hidden1result * input1
        h1weight2 += learning_rate * error_hidden1result * input2
        h1weight3 += learning_rate * error_hidden1result * input3
        h1weight4 += learning_rate * error_hidden1result * input4

        h2weight0 += learning_rate * error_hidden2result * biasNeuron
        h2weight1 += learning_rate * error_hidden2result * input1
        h2weight2 += learning_rate * error_hidden2result * input2
        h2weight3 += learning_rate * error_hidden2result * input3
        h2weight4 += learning_rate * error_hidden2result * input4


        # Calculating the new weights for between the hidden and output layers
        h_output1weight1 += learning_rate * error_output1 * h1output
        h_output1weight2 += learning_rate * error_output2 * h1output
        h_output1weight3 += learning_rate * error_output3 * h1output

        h_output2weight1 += learning_rate * error_output1 * h2output
        h_output2weight2 += learning_rate * error_output2 * h2output
        h_output2weight3 += learning_rate * error_output3 * h2output

        # Test data
        if selected_index == 24:
            selected_index = 50

        elif selected_index == 74:
            selected_index = 100

        if selected_index == 124:
            selected_index = 0
            epoch += 1

        else:
            selected_index += 1

        if epoch == 1000:
            break
    """
    # Training data
    for selected_index in range(50, 75):
        print("index: " + str(selected_index))
        input1 = iris.data[selected_index][0]
        input2 = iris.data[selected_index][1]
        input3 = iris.data[selected_index][2]
        input4 = iris.data[selected_index][3]
    
        print("Sepal Length: " + str(input1))
        print("Sepal Width: " + str(input2))
        print("Petal Length: " + str(input3))
        print("Petal Width: " + str(input4))
    
        print(iris.target_names[iris.target[selected_index]])
    """

    selected_index = 25
    correct = 0
    total = 0
    # Propagate through the testing data
    while selected_index < 150:
        input1 = iris.data[selected_index][0]
        input2 = iris.data[selected_index][1]
        input3 = iris.data[selected_index][2]
        input4 = iris.data[selected_index][3]

        h1inputSum = (biasNeuron * h1weight0) + (input1 * h1weight1) + (input2 * h1weight2) + \
                     (input3 * h1weight3) + (input4 * h1weight4)
        h2inputSum = (biasNeuron * h2weight0) + (input1 * h2weight1) + (input2 * h2weight2) + \
                     (input3 * h2weight3) + (input4 * h2weight4)

        # Sigmoid function to output from hidden layer
        h1output = sigmoid_function(h1inputSum)
        h2output = sigmoid_function(h2inputSum)

        # Try without bias for now (biasNeuron * h1weight0) +
        output1WeightedSum = (h1output * h_output1weight1) + (h2output * h_output2weight1)
        output2WeightedSum = (h1output * h_output1weight2) + (h2output * h_output2weight2)
        output3WeightedSum = (h1output * h_output1weight3) + (h2output * h_output2weight3)

        output1 = sigmoid_function(output1WeightedSum)
        output2 = sigmoid_function(output2WeightedSum)
        output3 = sigmoid_function(output3WeightedSum)

        target = iris.target[selected_index]
        real_output = get_output(output1, output2, output3)

        # Only test on the second half of the class data
        if selected_index == 50:
            selected_index = 74

        elif selected_index == 100:
            selected_index = 125

        if selected_index == 149:
            break

        else:
            selected_index += 1

        if target == real_output:
            correct += 1
        total += 1
    stop1 = time.time()
    runtime1 = stop1 - start1
    print(str(correct) + " correct guesses out of: " + str(total))
    print("5 - 2 - 3 topology accuracy: " + str(100 * correct / total) + " percent correct")
    print("Time elapsed is: " + str(runtime1))


# 5 - 6 - 3 topology

# Initialize the weights randomly
else:
    h1weight0 = random.uniform(-0.5, 0.5)
    h1weight1 = random.uniform(-0.5, 0.5)
    h1weight2 = random.uniform(-0.5, 0.5)
    h1weight3 = random.uniform(-0.5, 0.5)
    h1weight4 = random.uniform(-0.5, 0.5)

    h2weight0 = random.uniform(-0.5, 0.5)
    h2weight1 = random.uniform(-0.5, 0.5)
    h2weight2 = random.uniform(-0.5, 0.5)
    h2weight3 = random.uniform(-0.5, 0.5)
    h2weight4 = random.uniform(-0.5, 0.5)

    h3weight0 = random.uniform(-0.5, 0.5)
    h3weight1 = random.uniform(-0.5, 0.5)
    h3weight2 = random.uniform(-0.5, 0.5)
    h3weight3 = random.uniform(-0.5, 0.5)
    h3weight4 = random.uniform(-0.5, 0.5)

    h4weight0 = random.uniform(-0.5, 0.5)
    h4weight1 = random.uniform(-0.5, 0.5)
    h4weight2 = random.uniform(-0.5, 0.5)
    h4weight3 = random.uniform(-0.5, 0.5)
    h4weight4 = random.uniform(-0.5, 0.5)

    h5weight0 = random.uniform(-0.5, 0.5)
    h5weight1 = random.uniform(-0.5, 0.5)
    h5weight2 = random.uniform(-0.5, 0.5)
    h5weight3 = random.uniform(-0.5, 0.5)
    h5weight4 = random.uniform(-0.5, 0.5)

    h6weight0 = random.uniform(-0.5, 0.5)
    h6weight1 = random.uniform(-0.5, 0.5)
    h6weight2 = random.uniform(-0.5, 0.5)
    h6weight3 = random.uniform(-0.5, 0.5)
    h6weight4 = random.uniform(-0.5, 0.5)

    h_output1weight1 = random.uniform(-0.5, 0.5)
    h_output1weight2 = random.uniform(-0.5, 0.5)
    h_output1weight3 = random.uniform(-0.5, 0.5)

    h_output2weight1 = random.uniform(-0.5, 0.5)
    h_output2weight2 = random.uniform(-0.5, 0.5)
    h_output2weight3 = random.uniform(-0.5, 0.5)

    h_output3weight1 = random.uniform(-0.5, 0.5)
    h_output3weight2 = random.uniform(-0.5, 0.5)
    h_output3weight3 = random.uniform(-0.5, 0.5)

    h_output4weight1 = random.uniform(-0.5, 0.5)
    h_output4weight2 = random.uniform(-0.5, 0.5)
    h_output4weight3 = random.uniform(-0.5, 0.5)

    h_output5weight1 = random.uniform(-0.5, 0.5)
    h_output5weight2 = random.uniform(-0.5, 0.5)
    h_output5weight3 = random.uniform(-0.5, 0.5)

    h_output6weight1 = random.uniform(-0.5, 0.5)
    h_output6weight2 = random.uniform(-0.5, 0.5)
    h_output6weight3 = random.uniform(-0.5, 0.5)

    h1inputSum = 0

    # Learning rate
    learning_rate = 0.3

    # Initial epoch
    epoch = 1

    selected_index = 0
    start2 = time.time()
    while selected_index < 150:
        input1 = iris.data[selected_index][0]
        input2 = iris.data[selected_index][1]
        input3 = iris.data[selected_index][2]
        input4 = iris.data[selected_index][3]


        h1inputSum = (biasNeuron * h1weight0) + (input1 * h1weight1) + (input2 * h1weight2) + \
                     (input3 * h1weight3) + (input4 * h1weight4)
        h2inputSum = (biasNeuron * h2weight0) + (input1 * h2weight1) + (input2 * h2weight2) + \
                     (input3 * h2weight3) + (input4 * h2weight4)
        h3inputSum = (biasNeuron * h3weight0) + (input1 * h3weight1) + (input2 * h3weight2) + \
                     (input3 * h3weight3) + (input4 * h3weight4)
        h4inputSum = (biasNeuron * h4weight0) + (input1 * h4weight1) + (input2 * h4weight2) + \
                     (input3 * h4weight3) + (input4 * h4weight4)
        h5inputSum = (biasNeuron * h5weight0) + (input1 * h5weight1) + (input2 * h5weight2) + \
                     (input3 * h5weight3) + (input4 * h5weight4)
        h6inputSum = (biasNeuron * h6weight0) + (input1 * h6weight1) + (input2 * h6weight2) + \
                     (input3 * h6weight3) + (input4 * h6weight4)

        # Sigmoid function to output from hidden layer
        h1output = sigmoid_function(h1inputSum)
        h2output = sigmoid_function(h2inputSum)
        h3output = sigmoid_function(h3inputSum)
        h4output = sigmoid_function(h4inputSum)
        h5output = sigmoid_function(h5inputSum)
        h6output = sigmoid_function(h6inputSum)

        # Try without bias for now (biasNeuron * h1weight0) +
        output1WeightedSum = (h1output * h_output1weight1) + (h2output * h_output2weight1) + (h3output * h_output3weight1)
        + (h4output * h_output4weight1) + (h5output * h_output5weight1) + (h6output * h_output6weight1)
        output2WeightedSum = (h1output * h_output1weight2) + (h2output * h_output2weight2) + (h3output * h_output3weight2)
        + (h4output * h_output4weight2) + (h5output * h_output5weight2) + (h6output * h_output6weight2)
        output3WeightedSum = (h1output * h_output1weight3) + (h2output * h_output2weight3) + (h3output * h_output3weight3)
        + (h4output * h_output4weight3) + (h5output * h_output5weight3) + (h6output * h_output6weight3)

        output1 = sigmoid_function(output1WeightedSum)
        output2 = sigmoid_function(output2WeightedSum)
        output3 = sigmoid_function(output3WeightedSum)

        target1 = target_output(iris.target[selected_index], 0)
        target2 = target_output(iris.target[selected_index], 1)
        target3 = target_output(iris.target[selected_index], 2)

        error_output1 = output1 * (1 - output1) * (target1 - output1)
        error_output2 = output2 * (1 - output2) * (target2 - output2)
        error_output3 = output3 * (1 - output3) * (target3 - output3)

        error_hidden1 = (h_output1weight1 * error_output1) + (h_output1weight2 * error_output2) + (
                    h_output1weight3 * error_output3)
        error_hidden2 = (h_output2weight1 * error_output1) + (h_output2weight2 * error_output2) + (
                    h_output2weight3 * error_output3)
        error_hidden3 = (h_output3weight1 * error_output1) + (h_output3weight2 * error_output2) + (
                h_output3weight3 * error_output3)
        error_hidden4 = (h_output4weight1 * error_output1) + (h_output4weight2 * error_output2) + (
                h_output4weight3 * error_output3)
        error_hidden5 = (h_output5weight1 * error_output1) + (h_output5weight2 * error_output2) + (
                h_output5weight3 * error_output3)
        error_hidden6 = (h_output6weight1 * error_output1) + (h_output6weight2 * error_output2) + (
                h_output6weight3 * error_output3)

        error_hidden1result = error_hidden1 * (h1output * (1 - h1output))
        error_hidden2result = error_hidden2 * (h2output * (1 - h2output))
        error_hidden3result = error_hidden2 * (h3output * (1 - h3output))
        error_hidden4result = error_hidden2 * (h4output * (1 - h4output))
        error_hidden5result = error_hidden2 * (h5output * (1 - h5output))
        error_hidden6result = error_hidden2 * (h6output * (1 - h6output))

        # Updating weights through backpropagation
        h1weight0 += learning_rate * error_hidden1result * biasNeuron
        h1weight1 += learning_rate * error_hidden1result * input1
        h1weight2 += learning_rate * error_hidden1result * input2
        h1weight3 += learning_rate * error_hidden1result * input3
        h1weight4 += learning_rate * error_hidden1result * input4

        h2weight0 += learning_rate * error_hidden2result * biasNeuron
        h2weight1 += learning_rate * error_hidden2result * input1
        h2weight2 += learning_rate * error_hidden2result * input2
        h2weight3 += learning_rate * error_hidden2result * input3
        h2weight4 += learning_rate * error_hidden2result * input4

        h3weight0 += learning_rate * error_hidden3result * biasNeuron
        h3weight1 += learning_rate * error_hidden3result * input1
        h3weight2 += learning_rate * error_hidden3result * input2
        h3weight3 += learning_rate * error_hidden3result * input3
        h3weight4 += learning_rate * error_hidden3result * input4

        h4weight0 += learning_rate * error_hidden4result * biasNeuron
        h4weight1 += learning_rate * error_hidden4result * input1
        h4weight2 += learning_rate * error_hidden4result * input2
        h4weight3 += learning_rate * error_hidden4result * input3
        h4weight4 += learning_rate * error_hidden4result * input4

        h5weight0 += learning_rate * error_hidden5result * biasNeuron
        h5weight1 += learning_rate * error_hidden5result * input1
        h5weight2 += learning_rate * error_hidden5result * input2
        h5weight3 += learning_rate * error_hidden5result * input3
        h5weight4 += learning_rate * error_hidden5result * input4

        h6weight0 += learning_rate * error_hidden6result * biasNeuron
        h6weight1 += learning_rate * error_hidden6result * input1
        h6weight2 += learning_rate * error_hidden6result * input2
        h6weight3 += learning_rate * error_hidden6result * input3
        h6weight4 += learning_rate * error_hidden6result * input4

        # Calculating the new weights for between the hidden and output layers
        h_output1weight1 += learning_rate * error_output1 * h1output
        h_output1weight2 += learning_rate * error_output2 * h1output
        h_output1weight3 += learning_rate * error_output3 * h1output

        h_output2weight1 += learning_rate * error_output1 * h2output
        h_output2weight2 += learning_rate * error_output2 * h2output
        h_output2weight3 += learning_rate * error_output3 * h2output

        h_output3weight1 += learning_rate * error_output1 * h3output
        h_output3weight2 += learning_rate * error_output2 * h3output
        h_output3weight3 += learning_rate * error_output3 * h3output

        h_output4weight1 += learning_rate * error_output1 * h4output
        h_output4weight2 += learning_rate * error_output2 * h4output
        h_output4weight3 += learning_rate * error_output3 * h4output

        h_output5weight1 += learning_rate * error_output1 * h5output
        h_output5weight2 += learning_rate * error_output2 * h5output
        h_output5weight3 += learning_rate * error_output3 * h5output

        h_output6weight1 += learning_rate * error_output1 * h6output
        h_output6weight2 += learning_rate * error_output2 * h6output
        h_output6weight3 += learning_rate * error_output3 * h6output

        # Test data
        if selected_index == 24:
            selected_index = 50

        elif selected_index == 74:
            selected_index = 100

        if selected_index == 124:
            selected_index = 0
            epoch += 1

        else:
            selected_index += 1

        if epoch == 1000:
            break


    selected_index = 25
    correct = 0
    total = 0
    # Propagate through the testing data
    while selected_index < 150:
        input1 = iris.data[selected_index][0]
        input2 = iris.data[selected_index][1]
        input3 = iris.data[selected_index][2]
        input4 = iris.data[selected_index][3]

        h1inputSum = (biasNeuron * h1weight0) + (input1 * h1weight1) + (input2 * h1weight2) + \
                     (input3 * h1weight3) + (input4 * h1weight4)
        h2inputSum = (biasNeuron * h2weight0) + (input1 * h2weight1) + (input2 * h2weight2) + \
                     (input3 * h2weight3) + (input4 * h2weight4)

        # Sigmoid function to output from hidden layer
        h1output = sigmoid_function(h1inputSum)
        h2output = sigmoid_function(h2inputSum)

        # Try without bias for now (biasNeuron * h1weight0) +
        output1WeightedSum = (h1output * h_output1weight1) + (h2output * h_output2weight1)
        output2WeightedSum = (h1output * h_output1weight2) + (h2output * h_output2weight2)
        output3WeightedSum = (h1output * h_output1weight3) + (h2output * h_output2weight3)

        output1 = sigmoid_function(output1WeightedSum)
        output2 = sigmoid_function(output2WeightedSum)
        output3 = sigmoid_function(output3WeightedSum)

        target = iris.target[selected_index]
        real_output = get_output(output1, output2, output3)

        if selected_index == 50:
            selected_index = 74

        elif selected_index == 100:
            selected_index = 125

        if selected_index == 149:
            break

        else:
            selected_index += 1

        if target == real_output:
            correct += 1
        total += 1

    stop2 = time.time()
    runtime2 = stop2 - start2
    print(str(correct) + " correct guesses out of: " + str(total))
    print("5 - 6 - 2 topology accuracy: " + str(100 * correct / total) + " percent correct")
    print("Time elapsed: " + str(runtime2))