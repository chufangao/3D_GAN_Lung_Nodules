# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:03:55 2017

@author: SMAHESH
"""

#Code to make augmented positive and negative examples


import pickle
import gc
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
random.seed(10)

#global / box sizes (40,40,18)
Xsize = 40
Ysize = 40
Zsize = 18

counter = 0
counterzeta = 0

print ("Start")
slicefail = False

def createSample(listp, dictp, zcenter):
    listToReturn = []
    slicesfound = 0 #for debugging
    #print ("First item in dict:")
    centerZIndex = None
    if zcenter in dictp:
        centerZIndex = list(dictp.keys()).index(zcenter)
    minZIndex = int(centerZIndex - Zsize/2)
    maxZIndex = int(centerZIndex + Zsize/2)
    if minZIndex < 0 or maxZIndex > len(dictp): #for debugging
        print("Slice out of range")
        slicefail = True
        return ([], [])
    zlist = list(dictp.items())
    minzbound = float(zlist[minZIndex][0])
    maxzbound = float(zlist[maxZIndex][0])
    for j in range(minZIndex, maxZIndex):
        value = zlist[j][1]
        part = np.array(value)
        slicewanted = part[listp[0][0]:listp[0][1],listp[1][0]:listp[1][1]]
        listToReturn.append(slicewanted)
    return (listToReturn, [minzbound, maxzbound])


def CreateTranslatedPositive(listp, dictp, zcenter):
    #boxXY, imageDict, centerZ
    #Randomized translation:
    xShift = random.randint(-10, 10)
    yShift = random.randint(-10, 10)
    zShift = random.randint(-4, 4)
    
    listp[0][0] += xShift
    listp[0][1] += xShift
    listp[1][0] += yShift
    listp[1][1] += yShift
    
    listToReturn = []
    slicesfound = 0 #for debugging
    #print ("First item in dict:")
    centerZIndex = None
    if zcenter in dictp:
        centerZIndex = list(dictp.keys()).index(zcenter)
        centerZIndex += zShift
    minZIndex = int(centerZIndex - Zsize/2)
    maxZIndex = int(centerZIndex + Zsize/2)
    if minZIndex < 0 or maxZIndex > len(dictp): #for debugging
        print("Slice out of range")
        slicefail = True
        return ([], [])
    zlist = list(dictp.items())
    minzbound = float(zlist[minZIndex][0])
    maxzbound = float(zlist[maxZIndex][0])
    for k in range(minZIndex, maxZIndex):
        value = zlist[k][1]
        part = np.array(value)
        slicewanted = part[listp[0][0]:listp[0][1],listp[1][0]:listp[1][1]]
        listToReturn.append(slicewanted)
    return (listToReturn, [minzbound, maxzbound])


#functions
def createNegative(listp, dictp, slthick):
    #listp = exclude_set, dictp = imageDict, slice thickness
    #print("Negfunction")
    while (True):
        xmin = random.randint(0, 512 - Xsize)
        ymin = random.randint(0, 512 - Ysize)
        allZs = list(dictp.keys())
        validZs = allZs[int(0 + .5*Zsize) : int(len(allZs) - (.5*Zsize))]
        zcenter = random.choice(validZs)
        zmin = zcenter - slthick * .5 * Zsize 
        intersect = False
        for box in listp:
            xcoords = box[0]
            ycoords = box[1]
            zcoords = box[2]
            #print ("Neg Func")
            #print (zcoords)
            if xmin in range(xcoords[0] - Xsize, xcoords[1]):
                if ymin in range(ycoords[0] - Ysize, ycoords[1]):
                    if zmin >= zcoords[0] - slthick * Zsize and zmin <= zcoords[1]:
                        intersect = True
                        break                  
        if (not intersect):
            samplex = [xmin, xmin + Xsize]
            sampley = [ymin, ymin + Ysize]
            inputBox = [samplex, sampley]
            return createSample(inputBox, dictp, zcenter)


#init main
#load database information
fileextnesion = '.pickle'
loadedData = None

#load excel info
x1 = pd.ExcelFile("noduleDimensions.xlsx")
allNodules = x1.parse(x1.sheet_names[0])
#allNodules = df[ df["SliceThickness"] <= 2.5] df = df.drop(df[df.score < 50].index)
allNodules = allNodules.sort_values(['SeriesID'])
#print(allNodules)

IDs = list(allNodules["SeriesID"].drop_duplicates())
# validation_set = IDs[-120:-1]
# validation_set.append(IDs[-1])
validation_set = set()

nodulesToUse = x1.parse(x1.sheet_names[2])  
noduleSet = set(nodulesToUse["NoduleID"])
del nodulesToUse
del x1

#headerList = list(allNodules)
prevID = None
exclude_set = []
slicethickness = None
positivelist = []
sfailedposlist  = []
pfailedposlist  = []
negativelist = []
sfailedneglist = []
pfailedneglist = []
tempdict = None
imageDict = None
takeNegativeSample = True
seriesIDset = set()
print (len(allNodules))
counterx = 0
#iterate thru nodules w/ repetitions of ids
for i in range(len(allNodules)):
    if allNodules["SliceThickness"][i] <= 2.5:
        nodeID = allNodules["NoduleID"][i]
        seriesID = allNodules["SeriesID"][i]
        # if seriesID wasn't the last one, and not first element, and we want to take negative samples
        if prevID != seriesID and i != 0 and takeNegativeSample:
            for theta in range(10): #specifies how many negative samples we want per scan
                counterx += 1
                negtoadd, zholder = createNegative(exclude_set, imageDict, slicethickness)
                if (len(negtoadd) == Zsize):
                    negativelist.append(negtoadd)
                elif len((negtoadd)) == 0:
                    print ("Negsample Parse Failed: " + str(prevID))
                    pfailedneglist.append(prevID)
                else:
                    print ("Negsample Size Failed: " + str(prevID) + " was " + str(len(negtoadd)) + " slices")
                    sfailedneglist.append(prevID)   
            
        if prevID != seriesID:
            if seriesID in validation_set:
                print ("Validation Series ID: " + str(seriesID)) 
                break
            counter+=1
            if prevID in seriesIDset:
               print ("Repeate Series ID: " + str(prevID)) 
            else:
                seriesIDset.add(prevID)
            if counter % 100 == 0:
                print (str(counter) + " Done")  
            filestring = str(seriesID)
            with open(filestring, "rb") as f:
                loadedData = pickle.load(f)
            gc.collect()
            tempdict = loadedData
            imageDict = OrderedDict(sorted(tempdict.items(), key=lambda t: t[0]))
            slicethickness = allNodules["sliceDistances"][i]
            exclude_set = []
            takeNegativeSample = True

        #now onto pos examples?
        prevID = seriesID  
        if nodeID in noduleSet:
            #then store in data set
            centerX = allNodules["centerX"][i]
            boxX = [centerX-int(.5*Xsize), centerX+int(.5*Xsize)]
            centerY = allNodules["centerY"][i]
            boxY = [centerY-int(.5*Ysize), centerY+int(.5*Ysize)]
            centerZ = allNodules["centerZ"][i]
            #box [[x1,x2],[y1,y2]]
            boxXY = [boxX, boxY]
            postoadd = []
            
            if centerZ not in imageDict:
                centerZ *= -1
            if centerZ in imageDict:
                for w in range(7):
                    postoadd, boxZ = CreateTranslatedPositive(boxXY, imageDict, centerZ)
                    if slicefail:
                        print("Slices out of range: " + str(nodeID))
                        slicefail = False
                    elif (len(postoadd) == Zsize):
                        rnum = random.randint(-2, 0)
                        rnum = rnum * (-1)**random.randint(1, 3)
                        postoadd = np.dstack(postoadd)
                        if w == 2:
                            np.rot90(postoadd, rnum, axes = (0,1))
                        elif w == 3:
                            np.rot90(postoadd, rnum, axes = (0,2))
                        elif w == 4:
                            np.flipud(postoadd)
                        elif w == 5:
                            np.fliplr(postoadd)
                        elif w == 6:
                            np.rot90(postoadd, rnum, axes = (1,2))
                        elif w == 7:
                            if random.randint(0,2) == 1:
                                np.fliplr(postoadd)
                            else:
                                np.flipud(postoadd)
                                
                            axnum = random.randint(0,3)
                            if axnum == 0: 
                                np.rot90(postoadd, rnum, axes = (0,1))
                            elif axnum == 1:
                                np.rot90(postoadd, rnum, axes = (0,2))
                            else:
                                np.rot90(postoadd, rnum, axes = (1,2))
                        if postoadd.shape == (40,40,18):
                            positivelist.append(postoadd)
                        else:
                            print('nodeID faied: ', nodeID,'\nseriesID: ', seriesID)
                    elif len((postoadd)) == 0:
                        takeNegativeSample = False
                        pfailedposlist.append(nodeID)
                    else:
                        print ("Possample Size Failed: " + str(nodeID) + " was " + str(len(postoadd)) + " slices")
                        sfailedposlist.append(nodeID)
                        takeNegativeSample = False    
            
            else:
                takeNegativeSample = False
                pfailedposlist.append(nodeID)


                
        
        
        boxX = [allNodules["minimumX"][i], allNodules["maximumX"][i]]
        boxY = [allNodules["minimumY"][i], allNodules["maximumY"][i]]
        boxZ = [allNodules["minimumZ"][i], allNodules["maximumZ"][i]]
            
        centerZ = allNodules["centerZ"][i]
        if centerZ in imageDict:
            exclude_set.append([boxX, boxY, boxZ])
        elif centerZ * -1 in imageDict:
            temp = boxZ[0]
            boxZ[0] = -1 * boxZ[1]
            boxZ[1] = -1 * temp
            exclude_set.append([boxX, boxY, boxZ])
        else:
            takeNegativeSample = False
            print('centerz', centerZ)
            counterzeta += 1


print ("Neg Parse fails: " + str(len(pfailedneglist)))
print ("Neg Size fails: " + str(len(sfailedneglist)))
print ("Pos Parse fails: " + str(len(pfailedposlist)))
print ("Pos Size fails: " + str(len(sfailedposlist)))

print ("Positive samples: " + str(len(positivelist)))
print ("Negative samples: " + str(len(negativelist)))
print (counterx)
print (counterzeta)
with open("PositiveAugmented.pickle", 'wb') as handle:
    pickle.dump(positivelist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("NegativeAugmented.pickle", 'wb') as handle:
    pickle.dump(negativelist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("ValidationAugmented.pickle", 'wb') as handle:
    pickle.dump(validation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("PositiveerrorsAugmented.pickle", 'wb') as handle:
    pickle.dump(pfailedposlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sfailedposlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("NegativeerrorsAugmented.pickle", 'wb') as handle:
    pickle.dump(pfailedneglist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sfailedneglist, handle, protocol=pickle.HIGHEST_PROTOCOL)
