import train
import torch
import sys
import math

inputSize = train.n_digits
hiddenSize = train.n_hidden
outputSize = train.n_labels

def loadModel(filepath, inputSize, hiddenSize, outputSize):
    model = train.RNN(inputSize, hiddenSize, outputSize)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

# return output of an example
def evaluate(model, testInput):
    hidden = model.initHidden()

    for i in range(testInput.size()[0]):
        output, hidden = model(testInput[i], hidden)

    return output



if __name__ == '__main__':
    if (len(sys.argv) != 3) or (sys.argv[1] not in ["200", "2000", "20000"]):
        print("Usage: python test.py <number of training examples> <testFilepath>")
        sys.exit(1)

    testFilename = f"{sys.argv[2]}"
    testNumbers, testLabels = train.loadData(testFilename)
    testNumberTensors = train.numbersToTensors(testNumbers)

    predictions = []
    correctcount = 0
    totalcount = len(testNumbers)
    rnn = loadModel(f'trained_rnn_{sys.argv[1]}_examples.pth', inputSize, hiddenSize, outputSize)
    for i in range(len(testNumbers)):
        label = train.isPalindrome(testNumbers[i])
        testOutput = evaluate(rnn, testNumberTensors[i])
        topv, topi = testOutput.topk(1)
        predictValue = topv[0].item()
        predictLabel = topi[0].item()
        if predictLabel == label:
           correct = 'CORRECT'
           correctcount += 1
        else:
           correct =  'INCORRECT'
        print('%s (%.2f) %s %s' % (testNumbers[i], math.exp(predictValue), predictLabel, correct))
        predictions.append([int(testNumbers[i]), label, predictLabel, math.exp(predictValue), correct])

    
    print('-------------------------')
    print('Accuracy: %.4f \n' % (correctcount / totalcount)) # 0.71/0.67/0.71
    with open(f'predictions_{sys.argv[1]}.txt', 'w') as f:
        for number, label, pred, prob, correct in predictions:
            f.write(f"{number}\t{label}\t{pred}\t{prob}\t{correct}\n")



