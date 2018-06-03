import aleGraph
import pandas as pd

if __name__ == "__main__":
    #args
    t = 20  #minimum bigram frequency to keep in conflict graph
    eigen = 0.05
    cluster = 0.37
    pat_freq_thres = 100
    sample = -1         #<0 sample means it uses the full dataset
    #get Dataset to Train
    c_train = pd.read_csv('data/c_train.csv')
    nc_train = pd.read_csv('data/nc_train.csv')
    c_test = pd.read_csv('data/c_test.csv')
    nc_test = pd.read_csv('data/nc_test.csv')
    name = 'conflict_ale_20_0.05_0.37'

    # name = aleGraph.build_graph_and_model(c_train,nc_train,t,eigen,cluster,pat_freq_thres)
    # r = aleGraph.classify(c_test,nc_test,name,sample)
    # print(r)

    #Test has to be a sample of train, so I dont magically fit to result.
    aleGraph.batter_test_args(c_train, nc_train, 50, 100, 10, 3, 10, 1, 33, 40,1, 50, 100, 10, 1000)

