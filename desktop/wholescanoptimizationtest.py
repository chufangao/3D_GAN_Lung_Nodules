import numpy as np

def node_is_here(coords, boxes):
    for node in boxes:
        if coords in boxes[node]:
            return True
    return False

thresholds = [x/50.0 for x in list(range(1, 50))]
thresholds.extend([x/1000.0 for x in list(range(991, 1000))])
thresholds.extend([x/10000.0 for x in list(range(9991, 10000))])
thresholds.extend([x/100000.0 for x in list(range(99991, 100000))])
thresholds.extend([x/1000000.0 for x in list(range(999991, 1000000))])

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



nodulesFound = set()
fakeNodulesFound = set()
thresholds = np.sort(thresholds)
# thresholds = reversed(thresholds) # must reverse thresholds in order to avoid rebuilding nodulesFound

fpArr = np.zeros(len(thresholds))
tpArr = np.zeros(len(thresholds))
nodulesFoundArr = np.zeros(len(thresholds))
fakeNodulesFoundArr = np.zeros(len(thresholds))

for i in range(len(predictions)):
    threshold_index = thresholds.searchsorted(predictions[i][0], side='right')
    if threshold_index > 0:
        FP = True
        TP = False
        for node in noduleBoxes[seriesID]:
            if coords[i] in noduleBoxes[seriesID][node]:
                if not node in nodulesFound:
                    nodulesFound.add(node)
                    nodulesFoundArr[:threshold_index]+=1
                FP = False
                TP = True
        for node in fakeNoduleBoxes[seriesID]:
            if coords[i] in fakeNoduleBoxes[seriesID][node]:
                if not node in fakeNodulesFound:
                    fakeNodulesFound.add(node)
                    fakeNodulesFoundArr[:threshold_index]+=1
                FP = False
        if FP:
            fpArr[:threshold_index] += 1
        if TP:
            fpArr[:threshold_index] += 1
experimentDict[modelfile]['numDetected'] += nodulesFoundArr
experimentDict[modelfile]['numFakesDetected'] += nodulesFoundArr
experimentDict[modelfile]['sumofFPs'] += fpArr
experimentDict[modelfile]['sumofTPs'] += fpArr
