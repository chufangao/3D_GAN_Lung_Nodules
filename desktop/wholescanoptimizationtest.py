import numpy as np

def node_is_here(coords, boxes):
    for node in boxes:
        if coords in boxes[node]:
            return True
    return False

# thresholds = [x/50.0 for x in list(range(1, 50))]
# thresholds.extend([x/1000.0 for x in list(range(991, 1000))])
# thresholds.extend([x/10000.0 for x in list(range(9991, 10000))])
# thresholds.extend([x/100000.0 for x in list(range(99991, 100000))])
# thresholds.extend([x/1000000.0 for x in list(range(999991, 1000000))])

# predictions = np.random.uniform(size = 9999)
#
# for num in range(len(thresholds)):
#     FPs = 0
#     TPs = 0
#     nodulesFound = set()
#     fakeNodulesFound = set()
#
#     # for each threshold "num" iterate over every prediction "i"
#     for i in range(len(predictions)):
#         if predictions[i][0] >= thresholds[num]:
#             detection = coords[i]
#             FP = True
#             TP = False
#             # noduleBoxes is map from seriesIDs to a map from nodules to a list of overlapping coordinates
#             # this for(if()) thing basic just checks to see if there's a nodule at coords[i]
#             for node in noduleBoxes[seriesID]:
#                 if detection in noduleBoxes[seriesID][node]:
#                     nodulesFound.add(node)
#                     FP = False
#                     TP = True
#                     #can we break here?
#             #previously the code to check if the nodule was legitimate was here
#             #for optimization purposes I'm ignoring it for now
#             if FP:
#                 FPs += 1
#             if TP:
#                 TPs += 1
#     experimentDict[modelfile]['numDetected'][num] += len(nodulesFound)
#     experimentDict[modelfile]['numFakesDetected'][num] += len(fakeNodulesFound)
#     experimentDict[modelfile]['sumofFPs'][num] += FPs
#     experimentDict[modelfile]['sumofTPs'][num] += TPs



# # the set of all nodules detected at a given threshold
# nodulesFound = set()
# fakeNodulesFound = set()
# thresholds = np.sort(thresholds)
# thresholds = reversed(thresholds)
#
# fp_arr = np.zeros(len(thresholds))
# tp_arr = np.zeros(len(thresholds))
#
# for num in range(len(thresholds)):
#     # for each threshold "num" iterate over every prediction "i"
#     for i in range(len(predictions)):
#         threshold_index = thresholds.searchsorted(predictions[i][0], side='right')
#         #if predictions[i][0] >= thresholds[num]:
#         if threshold_index > 0:
#             FP = True
#             TP = False
#             for node in noduleBoxes[seriesID]:
#                 if coords[i] in noduleBoxes[seriesID][node]:
#                     nodulesFound.add(node)
#                     FP = False
#                     TP = True
#             for node in fakeNoduleBoxes[seriesID]:
#                 if coords[i] in fakeNoduleBoxes[seriesID][node]:
#                     fakeNodulesFound.add(node)
#                     FP = False
#             if FP:
#                 fp_arr[:threshold_index] += 1
#             if TP:
#                 fp_arr[:threshold_index] += 1
#     experimentDict[modelfile]['numDetected'][num] += len(nodulesFound)
#     experimentDict[modelfile]['numFakesDetected'][num] += len(fakeNodulesFound)
# experimentDict[modelfile]['sumofFPs'] += fp_arr
# experimentDict[modelfile]['sumofTPs'] += fp_arr

# the set of all nodules detected at a given threshold

thresholds = [x/100+.9 for x in range(10)]

experimentDict ={}
experimentDict['sumofFPs'] = [0 for i in range(len(thresholds))]
experimentDict['sumofTPs'] = [0 for i in range(len(thresholds))]
experimentDict['numDetected'] = [0 for i in range(len(thresholds))]
experimentDict['numFakesDetected'] = [0 for i in range(len(thresholds))]

allNoduleBoxes = np.unique(np.round(np.random.random((100,2)), decimals = 1), axis=0)
np.random.shuffle(allNoduleBoxes)
print(allNoduleBoxes.shape)
noduleBoxes = allNoduleBoxes[:allNoduleBoxes.shape[0]//2]
fakeNoduleBoxes = allNoduleBoxes[allNoduleBoxes.shape[0]//2:]

coords = [x/10 for x in range(0,11)]
print('coords'+str(coords))
predictions = np.random.random(len(coords))
print('predictions'+str(predictions))

nodulesFound = set()
fakeNodulesFound = set()
thresholds = np.sort(thresholds)
# thresholds = reversed(thresholds) # must reverse thresholds in order to avoid rebuilding nodulesFound

fpArr = np.zeros(len(thresholds))
tpArr = np.zeros(len(thresholds))
nodulesFoundArr = np.zeros(len(thresholds))
fakeNodulesFoundArr = np.zeros(len(thresholds))

for i in range(len(predictions)):
    threshold_index = thresholds.searchsorted(predictions[i], side='right')
    if threshold_index > 0:
        FP = True
        TP = False
        for node in noduleBoxes:
            if coords[i] == node[1]:
                if not node[0] in nodulesFound:
                    nodulesFound.add(node[0])
                    nodulesFoundArr[:threshold_index]+=1
                FP = False
                TP = True
        for node in fakeNoduleBoxes:
            if coords[i] == node[1]:
                if not node[0] in fakeNodulesFound:
                    fakeNodulesFound.add(node[0])
                    fakeNodulesFoundArr[:threshold_index]+=1
                FP = False
        if FP:
            fpArr[:threshold_index] += 1
        if TP:
            tpArr[:threshold_index] += 1
experimentDict['numDetected'] += nodulesFoundArr
experimentDict['numFakesDetected'] += fakeNodulesFoundArr
experimentDict['sumofFPs'] += fpArr
experimentDict['sumofTPs'] += tpArr





experimentDict2 ={}
experimentDict2['sumofFPs'] = [0 for i in range(len(thresholds))]
experimentDict2['sumofTPs'] = [0 for i in range(len(thresholds))]
experimentDict2['numDetected'] = [0 for i in range(len(thresholds))]
experimentDict2['numFakesDetected'] = [0 for i in range(len(thresholds))]

for num in range(len(thresholds)):
    FPs = 0
    TPs = 0
    nodulesFound = set()
    fakeNodulesFound = set()

    # for each threshold "num" iterate over every prediction "i"
    for i in range(len(predictions)):
        if predictions[i] >= thresholds[num]:
            detection = coords[i]
            FP = True
            TP = False
            for node in noduleBoxes:
                if detection == node[1]:
                    nodulesFound.add(node[0])
                    FP = False
                    TP = True
            for node in fakeNoduleBoxes:
                if detection == node[1]:
                    fakeNodulesFound.add(node[0])
                    FP = False
            if FP:
                FPs += 1
            if TP:
                TPs += 1
    experimentDict2['numDetected'][num] += len(nodulesFound)
    experimentDict2['numFakesDetected'][num] += len(fakeNodulesFound)
    experimentDict2['sumofFPs'][num] += FPs
    experimentDict2['sumofTPs'][num] += TPs

print('numDetected:' + str(experimentDict['numDetected']==experimentDict2['numDetected']))
print('dict1 numDetected:' + str(experimentDict['numDetected']))
print('dict2 numDetected:' + str(experimentDict2['numDetected']))

print('numFakesDetected:' + str(experimentDict['numFakesDetected']==experimentDict2['numFakesDetected']))
print('dict1 numFakesDetected:' + str(experimentDict['numFakesDetected']))
print('dict2 numFakesDetected:' + str(experimentDict2['numFakesDetected']))

print('sumofFPs:' + str(experimentDict['sumofFPs']==experimentDict2['sumofFPs']))
print('dict1 sumofFPs:' + str(experimentDict['sumofFPs']))
print('dict2 sumofFPs:' + str(experimentDict2['sumofFPs']))

print('sumofTPs:' + str(experimentDict['sumofTPs']==experimentDict2['sumofTPs']))
print('dict1 sumofTPs:' + str(experimentDict['sumofTPs']))
print('dict2 sumofTPs:' + str(experimentDict2['sumofTPs']))