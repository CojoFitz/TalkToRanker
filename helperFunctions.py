import numpy as np
import pandas as pd
import re, math
from matchResponse import matchResponse

def processHistogram(df, feats, binEdges):
  hisDf = pd.DataFrame()
  for i in range((len(feats))):
    df_fil = df[df.feature == feats[i]+'_v']
    counts, bins = np.histogram(df_fil["value"], bins=binEdges[i])
    binnedDf = pd.DataFrame({"bins": bins[1:], "counts": counts})
    binnedDf['feature'] = feats[i]
    hisDf = pd.concat([hisDf, binnedDf], ignore_index=True)
  return hisDf

def getBins (df, feats):
    binEdges = []
    for i in feats:
        df_fil = df[df.feature == i + '_v']
        featEdge = np.histogram_bin_edges(df_fil['value'], bins='scott')
        binEdges.append(featEdge)
    return binEdges


def suggest_cutoff(user_input_k, scores):
    n = len(scores)
    scores = np.sort(scores)[::-1]  # Sort in descending order
    if user_input_k >= n:
        return None
    k = user_input_k
    std_gap = (scores.max() - scores.min() ) /n
    while  k < n - 1:
        if scores[k] - scores[k+1] > std_gap:
            if(k == user_input_k):
                return None
            return [f'Warning: The score distances between {user_input_k} and {user_input_k +1} is small and hard to differentiate.', f'Try taking cutoff {k} instead. Since the distance between {k} and {k+1} is larger than stable score gap.']
        else:
            k+=1
    return None

def checkAttribution(checkDf):
    checkDf = checkDf.loc[:, checkDf.columns.str.endswith('_a')]   
    checkDf = checkDf.abs()
    avgs = checkDf.mean(axis=0)
    df_avgs = pd.DataFrame(avgs).reset_index()
    df_avgs.columns = ['feature', 'mean absolute attribution']
    df_avgs = df_avgs.sort_values(by=['mean absolute attribution'], axis=0, ascending=True)
    return df_avgs

def getFeatureList(df):
    allFeats = {
        'numerical' :[],
        'categorical' :  [],
        'all' : []
    }

    for col in df.columns:
        if(not col.endswith('_a') and col not in ['id','y']):
            if(col.endswith('_v')):
                formattedFeat = col[:-2]
                if(len(np.unique(df[col])) < 20): #Maybe add a check to see if the type of thbe column is numerical or not
                    allFeats['categorical'].append(formattedFeat)
                else:
                    allFeats['numerical'].append(formattedFeat)
                allFeats['all'].append(formattedFeat)
            else:
                allFeats['categorical'].append(col)
                allFeats['all'].append(col)
    if(len(allFeats['categorical']) == 0):
        allFeats['categorical'] = ['None']
    return allFeats

def prepareDF(df, selectionType, showFeatures): 
    #This formats the DF in a way that works best for the graph
    newDf = pd.melt(df, id_vars=['y','id'], var_name='feature', value_name='value')
    featuresUsed = [feature + "_v" for feature in showFeatures]
    newDf = newDf[newDf['feature'].isin(featuresUsed)]
    newDf['Selection'] = selectionType
    return newDf

def fairnessCalc(feature, df):
    if feature+'_v' in df.columns:
        feature = feature+'_v'
    return getRatio(feature,df)

def getRatio(feature, df):
    if feature+'_v' in df.columns:
        feature = feature+'_v'
    cats = df[feature].unique()
    ratioDict = {}
    for cat in cats:
        selectedCount = len(df[(df[feature] == cat) & (df['selection'] == 'selected')])
        ratio = selectedCount/len(df[feature]==cat)
        ratioDict[str(cat)] = ratio
    return ratioDict
    
def getAvgRatio(ratio):
  avgRatio = [np.average(ratio)]*len(ratio)
  return avgRatio

def crossEntropy(ratio):
  avgRatio = getAvgRatio(ratio)
  cross_entropy = 0
  for p, q in  zip (avgRatio, ratio):
        cross_entropy -= p * math.log(q)
  return cross_entropy

def suggestFairness(feature, df):
    df = df.sort_values(by=['y'], ascending=False, ignore_index=True) #Sort VALUES
    numEntries = len(df) #Get the number of entries
    maxEntropy = 0
    startingPoint = df.index[df['selection'] == 'selected'][-1]+1
    bestCutoff = startingPoint - 1 #Best cutoff is the cutoff amount that is suggested. We 
    endingPont = startingPoint + round( df[df['selection'] == 'selected'].shape[0] * 0.2)
    if(endingPont > numEntries):
        endingPont = numEntries
    for i in range(startingPoint, endingPont):
        df.loc[i, 'selection'] = 'selected'
        curRatio = getRatio(feature,df)
        ratList = list(curRatio.values())
        entropy = crossEntropy(ratList)
        if(entropy > maxEntropy):
            maxEntropy = entropy
            bestCutoff = i
    return bestCutoff


        
        

  
