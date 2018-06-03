import re
import nltk
from copy import deepcopy
import networkx as nx
import pandas as pd
import multiprocessing as mp
import multiprocessing
import math
import csv
from time import sleep

def preprocess_text(posts):
    text = str(posts['post_title'])+' . '+ str(posts['post_text'])+' .'
    text = text.lower()
    text =  re.sub('tl[;]?dr','',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+[0-9]+[s]?[ /\(,)]*f[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+[0-9]+[s]?[ /\(,)]*m[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+f[ /\(,)]*[0-9]+[s]?[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+m[ /\(,)]*[0-9]+[s]?[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[0-9]+','NUM',text,flags=re.IGNORECASE)
    text = re.sub('u/[^\s]+','AT_USER',text,flags=re.IGNORECASE)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text,flags=re.IGNORECASE)  #Convert www.* or https?://* to <url>
    text = text.split("[.]?\n[\* \[\(/]*[eE]dit")[0]
    text = text.split("[.]?\n[\* \[\(/]*EDIT")[0]
    text = text.split("[.]?\n[\* \[\(/]*big edit")[0]
    text = text.split("[.]?\n[\* \[\(/]*important edit")[0]
    text = text.split("[.]?\n[\* \[\(/]*[uU]pdate")[0]
    text = text.split("[.]?\n[\* \[\(/]*UPDATE")[0]
    text = text.split("[.]?\n[\* \[\(/]*big update")[0]
    text = text.split("[.]?\n[\* \[\(/]*important update")[0]
    text = text.split("[.]?\nfor an update")[0]
    text = text.replace('\r', '')
    return text

def bigram_frequency(posts):
    corpus = posts.str.cat(sep='. ')
    tokens = nltk.word_tokenize(corpus)
    bigrm = nltk.bigrams(tokens)
    fdist = nltk.FreqDist(bigrm)
    #normalize
    max_freq = fdist.get(fdist.max())
    return {k:v/max_freq for k,v in fdist.items()}

def get_freq(df):
    return bigram_frequency(df.apply(preprocess_text, axis=1))

def aggregate_freq(arg_c_freq,arg_nc_freq):
    res_freq = deepcopy(arg_c_freq)
    for k,nc_v in arg_nc_freq.items():
        if res_freq.get(k) is not None:
            c_v = res_freq.get(k)
            if c_v <= nc_v:
                res_freq.pop(k)   #not conflict has higher frequency so we dont delete the edge
            else:
                res_freq[k]= c_v - nc_v
    full_c_freq = deepcopy(res_freq)
    for k,c_v in full_c_freq.items():
        if c_v == 1:
            res_freq.pop(k)
    return res_freq

def create_graph(fdist,thresh):
    val = fdist[min(fdist, key=fdist.get)] * thresh
    fdist = {k: v for k, v in fdist.items() if v>val}
    nodes = set()
    edges = []
    for k,v in fdist.items():
        lst = list(k)
        [nodes.add(word) for word in lst]
        lst.append(v)
        edges.append(lst)
    DG=nx.DiGraph()
    DG.add_nodes_from(nodes)
    DG.add_weighted_edges_from(edges)
    return DG

def typify_graph(args_c_graph,eigen=0.02,cluster=0.3):
    res_c_graph = args_c_graph
    eigen_c = nx.eigenvector_centrality(res_c_graph)
    eigen_df = pd.DataFrame.from_dict(eigen_c, orient='index')
    nodes = list(res_c_graph)
    nodes_to_delete = pd.DataFrame(nodes,columns=['node'])
    eigen_df.columns = ['eigen_centrality']
    eigen_df['node'] = eigen_df.index
    context_words = eigen_df[eigen_df['eigen_centrality']>=eigen]
    c_u_graph = res_c_graph.to_undirected()
    clustering_score = nx.clustering(c_u_graph)
    cluster_df = pd.DataFrame.from_dict(clustering_score, orient='index')
    cluster_df.columns = ['clustering_score']
    cluster_df['node'] = cluster_df.index
    conflict_words = cluster_df[cluster_df['clustering_score']>=cluster]
    both_words = context_words.merge(conflict_words,how='inner',on=['node'])
    both_words = both_words.loc[:,['node']]
    context_words = context_words.merge(both_words,how='left', on=['node'],indicator=True)
    context_words = context_words[context_words['_merge'] == 'left_only']
    conflict_words = conflict_words.merge(both_words,how='left', on=['node'],indicator=True)
    conflict_words = conflict_words[conflict_words['_merge'] == 'left_only']
    context_words['type'] = 1
    conflict_words['type'] = 2
    both_words['type']=3
    context_words = context_words.loc[:,['node','type']]
    conflict_words = conflict_words.loc[:,['node','type']]
    words = pd.concat([context_words,conflict_words],ignore_index=True)
    words = pd.concat([words,both_words],ignore_index=True)
    #delete nodes that are not context words or conflict words
    nodes_to_delete = nodes_to_delete.merge(words,how='left',on=['node'],indicator=True)
    nodes_to_delete = nodes_to_delete[nodes_to_delete['_merge']=='left_only']
    for num,name in nodes_to_delete.iterrows():
            res_c_graph.remove_node(name['node'])
    words.index = words['node']
    words = words.loc[:,['type']]
    words = words.to_dict()
    words = words['type']
    nx.set_node_attributes(res_c_graph,words,'type')
    return res_c_graph

def generate_patterns(c_graph):
    patterns = []
    for node in c_graph.nodes():
        current_type = c_graph.node[node]['type']
        for suc in c_graph.successors(node):
            suc_type = c_graph.node[suc]['type']
            if (((current_type == 2) & (suc_type != 2)) | ((current_type != 2) & (suc_type == 2))):
                patterns.append((node if current_type != 2 else '.+') + ' ' + (suc if suc_type != 2 else '.+'))
            if (((current_type == 1) & (suc_type == 3)) | ((current_type == 3) & (suc_type == 1))):
                patterns.append((node if current_type != 3 else '.+') + ' ' + (suc if suc_type != 3 else '.+'))
            if ((current_type == 3) & (suc_type == 3)):
                patterns.append(node + ' .+')
                patterns.append('.+ ' + suc)

            for suc2 in c_graph.successors(suc):
                suc2_type = c_graph.node[suc2]['type']
                mis_tri = True
                if (((current_type == 2) & (suc_type != 2) & (suc2_type != 2)) | (
                        (current_type != 2) & (suc_type == 2) & (suc2_type != 2)) | (
                        (current_type != 2) & (suc_type != 2) & (suc2_type == 2))):
                    patterns.append(
                        (node if current_type != 2 else '.+') + ' ' + (suc if suc_type != 2 else '.+') + ' ' + (
                        suc2 if suc2_type != 2 else '.+'))
                    mis_tri = False
                if (mis_tri) & (((current_type == 1) & (suc_type == 1) & (suc2_type == 3)) | (
                        (current_type == 1) & (suc_type == 3) & (suc2_type == 1)) | (
                        (current_type == 3) & (suc_type == 1) & (suc2_type == 1))):
                    patterns.append(
                        (node if current_type != 3 else '.+') + ' ' + (suc if suc_type != 3 else '.+') + ' ' + (
                        suc2 if suc2_type != 3 else '.+'))
                    mis_tri = False
                if (mis_tri) & ((current_type == 1) & (suc_type == 3) & (suc2_type == 3)):
                    patterns.append(node + ' .+ ' + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 1) & (suc2_type == 3)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 3) & (suc2_type == 1)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' .+ ' + suc2)
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 3) & (suc2_type == 3)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' .+ ' + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False

                if (node == 'we') & (suc2 == "'s"):
                    print(suc)
    return list(set(patterns))

def build_graph_and_model(conflict,not_conflict,t,eigen,cluster,pat_freq_thres):
    #get bigram normalized frequencies to build the graph
    print('normalizing bigram frequencies...')
    c_freq = get_freq(conflict)
    nc_freq = get_freq(not_conflict)
    print('aggregating frequencies... ')
    #aggregate frequencies to build conflict graph
    print('building conflict graph... ')
    c_graph = create_graph(aggregate_freq(c_freq, nc_freq),thresh=t)
    #get eigen vector and clustering score and filter c_graph
    print('getting important nodes... ')
    c_graph = typify_graph(c_graph,eigen,cluster)
    #bootstrap patterns
    print('bootstrapping patterns... ')
    pats =  generate_patterns(c_graph)
    print('total patterns bootstrapped:',len(pats))
    name_conflict = 'conflict_ale_'+str(t)+'_'+str(eigen)+'_'+str(cluster)+'_'+str(pat_freq_thres)
    name_not_conflict = 'not_conflict_ale_'+str(t)+'_'+str(eigen)+'_'+str(cluster)+'_'+str(pat_freq_thres)
    threadize(prepare_for_features(conflict),pats,name_conflict)
    threadize(prepare_for_features(not_conflict),pats,name_not_conflict)
    print('done getting features for patterns... ')
    build_models(name_conflict, name_not_conflict, pat_freq_thres, conflict, not_conflict)
    return name_conflict


def prepare_for_features(texts):
    new_texts = texts.apply(preprocess_text,axis=1)
    new_texts = new_texts + '. '
    new_texts = new_texts.apply(lambda x:nltk.word_tokenize(x))
    return new_texts

def threadize(og_df,pats,name):
    cores = multiprocessing.cpu_count() - 1
    total = len(pats)
    inc = math.ceil(total/cores)
    processes = [mp.Process(target=pattern_features, args=(og_df,pats[i*inc:(i*inc)+inc],name+'_t'+str(i),name)) for i in range(cores)]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()


def pattern_features(df,patterns,name,sec):
    total = len(patterns)
    count = 0
    inc = 300
    thres = inc
    print('start...',name,sec)
    patterns = pd.DataFrame(patterns,columns=['pattern'])
    result = pd.DataFrame(columns=['pattern','freq','doc_freq','div'])
    for num,pattern in patterns.iterrows():
        pat = pattern['pattern']
        tokens = pat.split(' ')
        not_wildcard = [a for a in tokens if a !='.+']
        pre_docs = df[df.apply(lambda x: not_wildcard[0] in x)]
        if len(not_wildcard) > 1:
            pre_docs = pre_docs[pre_docs.apply(lambda x: not_wildcard[1] in x)]
        #filtered, so now we go one by one finding the full sequence
        diversity = []
        freq = 0
        acum_doc = 0
        for doc in pre_docs:
            doc_has = False
            for i in range(len(doc)):
                if doc[i] == not_wildcard[0]:
                    if not_wildcard[0]==tokens[0]:  #the first word is an actual word
                        if len(tokens)==2:   #bigram
                            if (i+1) < len(doc):  #there is still space for the wildcard
                                diversity.append(doc[i+1])
                                freq += 1
                                doc_has = True
                        else:                #trigram
                            if (i+2) < len(doc):  #there is still space for the pattern
                                if not_wildcard[1] ==tokens[1]:  #second word is actual word
                                    if doc[i+1] == not_wildcard[1]:
                                        diversity.append(doc[i+2])
                                        freq += 1
                                        doc_has = True
                                elif doc[i+2] == not_wildcard[1]:   #third word is actual word
                                    diversity.append(doc[i+1])
                                    freq += 1
                                    doc_has = True
                    elif not_wildcard[0] == tokens[1]: #first word is wildcard
                        if (i-1) >= 0:
                            if len(tokens)==2:
                                diversity.append(doc[i-1])
                                freq += 1
                                doc_has = True
                            else:
                                if (i+1) < len(doc):
                                    if doc[i+1]==not_wildcard[1]:
                                        diversity.append(doc[i-1])
                                        freq += 1
                                        doc_has = True
            if doc_has:
                acum_doc += 1
        result= result.append({'pattern':pattern['pattern'],'freq':freq,'doc_freq':acum_doc,'div':len(set(diversity))}, ignore_index=True)
        count += 1
        if count > thres:
            print('done with ',count,'out of ',total,'of ',name)
            thres += inc
    result.to_csv('patterns/'+name+'.csv',index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
    done = pd.DataFrame([name])
    done['sec']=sec
    done.to_csv('controls/threads_done.csv',index=False,encoding='utf-8',mode='a',header=False)

def unify(name):
    result = pd.DataFrame()
    control = pd.read_csv('controls/threads_done.csv')
    control = control[control['sec']==name]
    for num,file in control.iterrows():
        cur_file = pd.read_csv('patterns/'+file['name']+'.csv')
        result = pd.concat([result,cur_file])
    return result


def c_score(pats, docs):
    if pats['doc_freq_c'] == 0:
        return 0
    freq_c = math.log(pats['freq_c'] + 1)
    doc_freq_c = math.log(1 + (docs / pats['doc_freq_c']))
    diversity_c = math.log(pats['div_c'] + 1)
    freq_nc = math.log(pats['freq_nc'] + 1)
    doc_freq_nc = math.log(1 + (docs / pats['doc_freq_nc']))
    diversity_nc = math.log(pats['div_nc'] + 1)

    return (freq_c * doc_freq_c * diversity_c) - (freq_nc * doc_freq_nc * diversity_nc)


def get_avg_len(df):
    pre_posts = prepare_for_features(df)
    x = pre_posts.apply(lambda row: len(row))
    return x.sum()/len(x)


def build_models(name_conflict,name_not_conflict,pat_freq_thres,conflict,not_conflict):
    print('starting building model...')
    c_degrees = unify(name_conflict)
    nc_degrees = unify(name_not_conflict)
    c_degrees = c_degrees.drop_duplicates()
    nc_degrees = nc_degrees.drop_duplicates()
    print('filtering those under pre-defined threshold...')
    c_degrees = c_degrees[c_degrees['freq'] >= pat_freq_thres]
    c_degrees = c_degrees.merge(nc_degrees, on='pattern', how='inner', suffixes=['_c', '_nc'])
    c_degrees['degree_c'] = c_degrees.apply(lambda x: c_score(x, len(conflict)), axis=1)
    c_degrees = c_degrees[c_degrees['degree_c'] > 0]
    just_degrees_c = c_degrees.loc[:, ['degree_c']]
    just_degrees_c = just_degrees_c.drop_duplicates()
    print('building ranks...')
    just_degrees_c['c_rank'] = just_degrees_c['degree_c'].rank(ascending=True)
    c_degrees = c_degrees.merge(just_degrees_c, on='degree_c')
    c_degrees = c_degrees.drop_duplicates()
    print('defining not conflict threshold and average length...')
    max_rank = c_degrees['c_rank'].max()
    avg_len = get_avg_len(not_conflict)
    x = c_degrees.apply(lambda x: x['freq_nc'] * math.pow(math.e, x['c_rank'] / max_rank), axis=1)
    x = pd.DataFrame([x.sum() / len(not_conflict)], columns=['nc_threshold'])
    x['avg_len'] = avg_len

    c_degrees.to_csv('data/c_model_'+name_conflict+'.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    x.to_csv('data/nc_threshold_'+name_conflict+'.csv', index=False, encoding='utf-8')
    print('models built:','data/c_model_'+name_conflict+'.csv','data/nc_threshold_'+name_conflict+'.csv')

def threadize_class(og_df,name,label):
    cores = multiprocessing.cpu_count() - 1
    total = len(og_df)
    inc = math.ceil(total/cores)
    if inc*cores > total:
        cores = math.ceil(total/inc)
    processes = [mp.Process(target=classify_w_model, args=(og_df[i*inc:(i*inc)+inc],name+'_t'+str(i),name,label)) for i in range(cores)]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()


def classify_w_model(posts, name, sec,label):
    print('starting...', name, sec,label)
    c_model = pd.read_csv('data/c_model_'+sec+'.csv')
    nc_thres = pd.read_csv('data/nc_threshold_'+sec+'.csv')
    avg_len = nc_thres.loc[0]['avg_len']
    nc_thres = nc_thres.loc[0]['nc_threshold']
    res_posts = deepcopy(posts)
    res_posts = res_posts.loc[:, ['post_id']]
    res_posts.index = range(len(res_posts.index))
    pre_posts = prepare_for_features(posts)
    f_result = pd.DataFrame(columns=['c_score', 'class', 'post_size'])
    count = 0
    inc = 40
    top = inc
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    for post in pre_posts:
        result = pd.DataFrame(columns=['pattern', 'freq'])
        for num, i in c_model.iterrows():
            pat = i['pattern']
            tokens = pat.split(' ')
            not_wildcard = [a for a in tokens if a != '.+']
            freq = 0
            con_all = True
            for wo in not_wildcard:
                if wo not in post:
                    con_all = False
            if con_all:
                positions = get_indexes(not_wildcard[0], post)
                for i in positions:
                    if post[i] == not_wildcard[0]:
                        if not_wildcard[0] == tokens[0]:  # the first word is an actual word
                            if len(tokens) == 2:  # bigram
                                if (i + 1) < len(post):  # there is still space for the wildcard
                                    freq += 1
                            else:  # trigram
                                if (i + 2) < len(post):  # there is still space for the pattern
                                    if not_wildcard[1] == tokens[1]:  # second word is actual word
                                        if post[i + 1] == not_wildcard[1]:
                                            freq += 1
                                    elif post[i + 2] == not_wildcard[1]:  # third word is actual word
                                        freq += 1
                        elif not_wildcard[0] == tokens[1]:  # first word is wildcard
                            if (i - 1) >= 0:
                                if len(tokens) == 2:
                                    freq += 1
                                else:
                                    if (i + 1) < len(post):
                                        if post[i + 1] == not_wildcard[1]:
                                            freq += 1
            result = result.append({'pattern': pat, 'c_freq': freq}, ignore_index=True)
        c_result = result.merge(c_model, how='inner', on=['pattern'])
        max_rank = c_model['c_rank'].max()
        x_res = c_result.apply(lambda x_res: x_res['c_freq'] * math.pow(math.e, x_res['c_rank'] / max_rank), axis=1)
        c_score = x_res.sum()

        # nc_threshold is variable depending on the post lenght analysed
        new_thresh = (nc_thres * len(post)) / avg_len
        res_class = 'c' if c_score > new_thresh else 'nc'
        f_result = f_result.append(
            {'c_score': c_score, 'class': res_class, 'post_size': len(post)}, ignore_index=True)
        count += 1
        if count > top:
            print('done with ', count, 'total', len(res_posts))
            top += inc
    f_result.index = range(len(f_result.index))
    f_result['label'] = label
    name_to_save = name + '_'+label
    f_result.join(res_posts).to_csv('classification/' + name_to_save + '.csv', index=False, encoding='utf-8',
                                    quoting=csv.QUOTE_NONNUMERIC)
    done = pd.DataFrame([name_to_save])
    done['sec'] = sec
    done.to_csv('controls/class_threads_done.csv', index=False, encoding='utf-8', mode='a', header=False)


def unify_class(name):
    result = pd.DataFrame()
    control = pd.read_csv('controls/class_threads_done.csv')
    control = control[control['sec']==name]
    for num,file in control.iterrows():
        cur_file = pd.read_csv('classification/'+file['name']+'.csv')
        result = pd.concat([result,cur_file])
    return result


def classify(c_test,nc_test,name,sample):
    if sample > 0:
        threadize_class(c_test.sample(sample), name, 'c')
        threadize_class(nc_test.sample(sample), name, 'nc')
    else:
        threadize_class(c_test, name, 'c')
        threadize_class(nc_test, name, 'nc')
    res_c = unify_class(name)
    res_c['correct'] = res_c.apply(lambda row: 1 if row['class'] == row['label'] else 0, axis=1)
    pre_c = res_c[res_c['label']=='c']
    pre_tf = res_c[res_c['class']=='c']
    accuracy = res_c['correct'].sum() / len(res_c)
    precision = pre_c['correct'].sum()/len(pre_c)
    recall = pre_c['correct'].sum()/len(pre_tf)
    F1 = 2*((precision*recall)/(precision+recall))
    result = pd.DataFrame([name],columns=['name'])
    result['accuracy']=accuracy
    result['precision']=precision
    result['recall']=recall
    result['F1']=F1
    return result



def batter_test_args(c_train,nc_train,t_min,t_max,t_inc,e_min,e_max,e_inc,c_min,c_max,c_inc,p_min,p_max,p_inc,sample):
    first = True
    mode = 'w'
    for t in range(t_min,t_max,t_inc):
        for e in range(e_min,e_max,e_inc):
            for c in range(c_min,c_max,c_inc):
                for p in range(p_min,p_max,p_inc):
                    name = build_graph_and_model(c_train,nc_train,t,e/100,c/100,p)
                    c_test = c_train.sample(sample)
                    nc_test = nc_train.sample(sample)
                    r = classify(c_test,nc_test,name,sample)
                    r['t']=t
                    r['e']=e/100
                    r['c']=c/100
                    r['p']=p
                    r.to_csv('classification/batter_results.csv', index=False, encoding='utf-8', mode=mode,header=first)
                    first = False
                    mode = 'a'
                    sleep(120)
