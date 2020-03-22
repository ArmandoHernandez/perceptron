import os
import sys
import getopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import http.server
import socketserver
import threading
import signal


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = self.request.recv(1024)
        cur_thread = threading.current_thread()
        response = "{}: {}".format(cur_thread.name, data)
        self.request.sendall(response)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass




def graph(lines, redPoints, bluePoints) :
    plt.scatter(redPoints['x'], redPoints['y'], color='red')
    plt.scatter(bluePoints['x'], bluePoints['y'], color='blue')
    plt.axis([-0.5, 1.5, -0.5, 1.5])

    for i in range(len(lines)) :
        plt.plot([lines[i][1][0], lines[i][0][0]+lines[i][1][0]], 'g--', linewidth=1)

    plt.xticks()
    plt.yticks()

    plt.show()
    plt.savefig('./www/perceptron_graph.png')


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b) :
    return stepFunction((np.matmul(X,W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate = 0.01) :
    for i in range(len(X)):
        y_hat = prediction(X.iloc[i],W,b)
        if (y[i]-y_hat == 1) :
                W[0] = W[0] + (X.iloc[i].T[0]*learn_rate)
                W[1] = W[1] + (X.iloc[i].T[1]*learn_rate)
                b += learn_rate
        elif (y[i]-y_hat == -1) :
                W[0] = W[0] - (X.iloc[i].T[0]*learn_rate)
                W[1] = W[1] - (X.iloc[i].T[1]*learn_rate)
                b -= learn_rate
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25) :
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + max(X.T[0])

    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs) :
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        # remove 'Unsued variable 'i' warning
        i=i

    return boundary_lines


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def main(argv):

    # by default, we take our own dataframe
    cvs_file = './data.csv'
    learn_rate = 0.01
    num_epochs = 25
    seed = 42
    serv_port = 8080


    # check that the given inputs comply with the input
    # arguments requirements
    try:
        opts, args = getopt.getopt(argv, "i:l:e:s:p:",
                        ["input-data=","learning-rate=","num-epochs=","seed=", "tcp-server-port="])
    except getopt.GetoptError:
        print("Incorrect ussage of arguments.")
        print("Correct ussage is:")
        print("     perceptron.py [-i <input_cvsfile>][-l <float learning rate>][-e <int number of epochs][-s <int seed>][-p <int port tcp server]")
        print("You can also substitute the flags as follows:")
        print("     --input-data        vs. -i")
        print("     --learning-rate     vs. -l")
        print("     --num-epochs        vs. -e")
        print("     --seed              vs. -s")
        print("     --tcp-server-port   vs. -p")
        sys.exit(2)

    # process the -possible- given arguments
    for opt, arg in opts:
        if opt in ("-i", "--input-data"):
            cvs_file = arg
            if not cvs_file.endswith('.csv'):
                print("Input file must be a csv file with three columns (i.e. x, y, class). Ej.")
                print("x,y,class")
                print("0.78051,-0.063669,1")
                print("0.28774,0.29139,1")
                print("0.40714,0.17878,1")
                print("......")
                sys.exit(2)
        elif opt in ("-l", "--learning-rate"):
            learn_rate = float(arg)
        elif opt in ("-e", "--num-epochs"):
            num_epochs = int(arg)
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-p", "--tcp-server-port"):
            serv_port = int(arg)
    # remove warning "Unsued variable 'args'"
    args=args


    # Setting the random seed, feel free to change it and see different solutions.
    np.random.seed(seed)

    data = pd.read_csv(cvs_file, encoding='utf-8')
    redPoints = pd.DataFrame()
    bluePoints = pd.DataFrame()
    X = data.drop('class', axis=1)
    y = data['class']

    for index, row in data.iterrows() :
        if(row['class'] == 1.0) :
            bluePoints = bluePoints.append(data.iloc[index])
        else :
            redPoints = redPoints.append(data.iloc[index])

    # train the perceptron and generate the graph of the solution
    graph( trainPerceptronAlgorithm(X, y,learn_rate=learn_rate,num_epochs=num_epochs), redPoints, bluePoints )

    # change to the www directory to start the server from there
    # (i.e. to serve that directory only)
    web_dir = os.path.join(os.path.dirname(__file__), 'www')
    os.chdir(web_dir)

    #handler = http.server.SimpleHTTPRequestHandler

    #with socketserver.TCPServer(("", serv_port), handler) as httpd:
    #    httpd.serve_forever()

    server = ThreadedTCPServer(("", serv_port), ThreadedTCPRequestHandler)
    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)
    print("serving at port ", serv_port)
    
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl-l to stop the socket server...")
    signal.pause()
    server.shutdown()
    server.server_close()



if __name__=="__main__":
    main(sys.argv[1:])