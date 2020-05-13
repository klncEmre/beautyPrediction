import PIL.Image
import PIL.ImageDraw
import numpy as np

from keras import models
from keras.layers import Dense
from keras.models import Sequential

from keras.optimizers import Adam,SGD

import math

import face_recognition
show = False
ratiosGeneral = []
ratiosLast = []
ratio_for_new = []


def get_the_Ratio(point1, point2, point3, point4):
    distance1 = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    distance2 = math.sqrt((point3[0] - point4[0]) ** 2 + (point3[1] - point4[1]) ** 2)
    return distance1 / distance2



def imageProcces(link, forNew):
    fr_image = face_recognition.load_image_file(link)

    face_landmarks_list = face_recognition.face_landmarks(fr_image)
    global ratiosLast
    global ratio_for_new
    global ratiosGeneral

    def distance(point1, point2):
        distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return distance

    pil_image = PIL.Image.fromarray(fr_image)
    list_of_data = []
    left_eye = face_landmarks_list[0]["left_eye"]
    right_eye = face_landmarks_list[0]["right_eye"]
    nose_bridge = face_landmarks_list[0]["nose_bridge"]
    nose_tip = face_landmarks_list[0]["nose_tip"]
    chin = face_landmarks_list[0]["chin"]
    l_eye_brow = face_landmarks_list[0]["left_eyebrow"]
    r_eye_brow = face_landmarks_list[0]["right_eyebrow"]
    t_lip = face_landmarks_list[0]["top_lip"]
    b_lip = face_landmarks_list[0]['bottom_lip']
    midofeyes = ((left_eye[0][0] + left_eye[3][0]) / 2), ((left_eye[3][1] + left_eye[0][1]) / 2)
    midofeyeR = ((right_eye[0][0] + right_eye[3][0]) / 2), ((right_eye[3][1] + right_eye[3][1]) / 2)


    averageDistanceOfCtoLTop = (distance(t_lip[0],chin[4]) + distance(t_lip[6],chin[12])) / 2
    averageDistanceOfCtoLBottom = (distance(t_lip[0],chin[6]) + distance(t_lip[6],chin[10])) / 2

    eye0toeyeBrow4 = (distance(left_eye[0],l_eye_brow[4]) + distance(right_eye[0],r_eye_brow[4]) )/ 2
    eye0toeye3 = distance(left_eye[0],left_eye[3])


    lefteyebrow0tolefteyebrow4 = (distance(l_eye_brow[0],l_eye_brow[4]) + distance(r_eye_brow[0],right_eye[4]))/2

    eye0toeyebrow0 = (distance(l_eye_brow[0],left_eye[0]) + distance(r_eye_brow[0],r_eye_brow[0]))/2
    eye3toeyeBrow4 = (distance(l_eye_brow[4],left_eye[3]) + distance(r_eye_brow[4],r_eye_brow[3]))/2
    # pointList = [l_eye_brow[0],l_eye_brow[4],r_eye_brow[0],right_eye[4],left_eye[0],left_eye[2],left_eye[3],left_eye[4],right_eye[0],right_eye[2],right_eye[3],right_eye[4],
    # nose_bridge[0],nose_bridge[3],nose_tip[0],nose_tip[2],nose_tip[4],t_lip[0],t_lip[3] ,
    # t_lip[6],t_lip[9],b_lip[0],b_lip[3],b_lip[6],b_lip[9],chin[0],chin[4],chin[8],chin[12],chin[16]]
    ratiosGeneral = []
    ratiosGeneral.append(averageDistanceOfCtoLBottom/averageDistanceOfCtoLTop/3)
    ratiosGeneral.append(get_the_Ratio(left_eye[3], right_eye[0], left_eye[0], left_eye[3]) / 5)  # ratio1 eye vs beetween eye
    ratiosGeneral.append(get_the_Ratio(midofeyes, nose_tip[2], nose_tip[2], chin[8]) / 5)
    ratiosGeneral.append(get_the_Ratio(t_lip[3], nose_tip[2], b_lip[3], chin[8]) / 5)
    ratiosGeneral.append(get_the_Ratio(left_eye[0], left_eye[3], t_lip[0], t_lip[6]) / 5)
    ratiosGeneral.append(get_the_Ratio(left_eye[4], left_eye[2], nose_bridge[0], nose_bridge[3]) / 5)
    ratiosGeneral.append(eye0toeyeBrow4/eye0toeye3)
    ratiosGeneral.append(get_the_Ratio(left_eye[0], right_eye[3], nose_bridge[0], nose_tip[2]) / 5)
    ratiosGeneral.append(get_the_Ratio(left_eye[3], right_eye[0], nose_tip[0], nose_tip[4]) / 5)
    ratiosGeneral.append(get_the_Ratio(left_eye[2], left_eye[4], b_lip[9], b_lip[3]) / 5)
    ratiosGeneral.append(get_the_Ratio(l_eye_brow[4], chin[8], left_eye[3], t_lip[0]) / 5)
    ratiosGeneral.append(get_the_Ratio(t_lip[3], t_lip[9], b_lip[9], b_lip[3]) / 5)
    ratiosGeneral.append(get_the_Ratio(nose_tip[0], nose_tip[4], nose_bridge[0], nose_bridge[3]) / 5)
    ratiosGeneral.append(get_the_Ratio(nose_tip[0], nose_tip[4], left_eye[0], left_eye[3]) / 5)
    ratiosGeneral.append((distance(nose_bridge[0],nose_bridge[3])*distance(t_lip[3],b_lip[3])/(distance(nose_bridge[0],nose_bridge[3])+distance(t_lip[3],b_lip[3]))))
    #ratiosGeneral.append(get_the_Ratio(l_eye_brow[0], l_eye_brow[4], t_lip[0], t_lip[6]) / 3)
    ratiosGeneral.append(lefteyebrow0tolefteyebrow4/distance(t_lip[0],t_lip[6])/4)
    #ratiosGeneral.append(get_the_Ratio(left_eye[2], l_eye_brow[4], left_eye[4], l_eye_brow[4]) / 5)
    ratiosGeneral.append(eye0toeyebrow0/eye3toeyeBrow4)

    eye2toEye4 = (distance(left_eye[2],left_eye[4]) + distance(right_eye[2],right_eye[4]))/2
    noset2Toeyebrow4 = (distance(nose_tip[2],l_eye_brow[0]) + distance(nose_tip[2],r_eye_brow[4]))/2
    ratiosGeneral.append(eye2toEye4/noset2Toeyebrow4)

    #ratiosGeneral.append(get_the_Ratio(left_eye[2], left_eye[4], nose_tip[2], l_eye_brow[3]) / 5)


    ratiosGeneral.append(get_the_Ratio(left_eye[0], right_eye[3], chin[7], chin[9]) / 5)
    ratiosGeneral.append(get_the_Ratio(left_eye[0], l_eye_brow[3], left_eye[2], left_eye[4]) / 5)
    ratiosGeneral.append(get_the_Ratio(midofeyes, midofeyeR, t_lip[0], t_lip[6]) / 5)
    ratiosGeneral.append(get_the_Ratio(midofeyes,midofeyeR,nose_tip[0],nose_tip[4]))
    ratiosGeneral.append(get_the_Ratio(b_lip[9],b_lip[3],chin[7],chin[9])/5)

    #ratiosGeneral.append(get_the_Ratio(right_eye[0],r_eye_brow[3],t_lip[3],b_lip[9])/5)
    eye0toeyeBrow3 = (distance(left_eye[0], l_eye_brow[1]) + distance(right_eye[0], r_eye_brow[3])) / 2
    tliptoBottomLip =(distance(t_lip[3],b_lip[9]) )
    ratiosGeneral.append(eye0toeyeBrow3/tliptoBottomLip)


    #ratiosGeneral.append(get_the_Ratio(nose_tip[2],l_eye_brow[3],left_eye[0],left_eye[2])/5)
    ntipToeyeBrow3 = (distance(nose_tip[2], l_eye_brow[1]) + distance(nose_tip[2], r_eye_brow[3])) / 2
    ratiosGeneral.append(ntipToeyeBrow3/eye0toeye3)

    ratiosGeneral.append(get_the_Ratio(chin[7],chin[8],b_lip[9],chin[8])/5)

    #ratiosGeneral.append(get_the_Ratio(left_eye[3],nose_bridge[3],nose_tip[2],t_lip[0])/5)
    eye3toNoseBridge3 = (distance(nose_bridge[3], left_eye[3]) + distance(nose_bridge[3], right_eye[3])) / 2
    nose_tip2Totlip9 =  (distance(nose_tip[2], t_lip[9]) )
    ratiosGeneral.append(eye3toNoseBridge3/nose_tip2Totlip9)



    ratiosGeneral.append(get_the_Ratio(left_eye[0],chin[8],nose_tip[2],nose_bridge[0])/5)

    #ratiosGeneral.append(get_the_Ratio(right_eye[0],nose_tip[4],t_lip[0],b_lip[0])/5)
    eye0ToTip4 = (distance(left_eye[0], nose_tip[4]) + distance(right_eye[3], nose_tip[0])) / 2
    leye0ToReye3 = (distance(left_eye[0], right_eye[3]))
    ratiosGeneral.append(eye0ToTip4/leye0ToReye3)
    ratiosGeneral.append(eye0ToTip4/tliptoBottomLip)
    ratiosGeneral.append(leye0ToReye3/tliptoBottomLip)


    ratiosGeneral.append(get_the_Ratio(nose_bridge[0],b_lip[0],nose_bridge[0],chin[8])/5)
    ratiosGeneral.append(get_the_Ratio(l_eye_brow[1],b_lip[0],nose_bridge[0],b_lip[3])/5)
    ratiosGeneral.append(get_the_Ratio(left_eye[4],left_eye[2],nose_bridge[0],nose_bridge[3])/5)
    ratiosGeneral.append(get_the_Ratio(left_eye[2],left_eye[4],nose_tip[0],nose_tip[4])/5)
    ratiosGeneral.append(get_the_Ratio(left_eye[4],l_eye_brow[3],nose_bridge[0],nose_bridge[3])/5)

    ratiosGeneral.append(get_the_Ratio(b_lip[9],chin[7],b_lip[9],chin[8])/5)
    ratiosGeneral.append(get_the_Ratio(left_eye[0],right_eye[3],nose_bridge[0],chin[8])/5)
    ratiosGeneral.append(get_the_Ratio(left_eye[2],left_eye[4],b_lip[3],b_lip[9])/5)
    ratiosGeneral.append(get_the_Ratio(chin[7],chin[8],b_lip[0],b_lip[6])/5)

    #ratiosGeneral.append(get_the_Ratio(l_eye_brow[3],nose_tip[4],nose_bridge[0],nose_bridge[3])/5)
    eyeB1toNoseTip4 = (distance(l_eye_brow[1],nose_tip[4])+distance(r_eye_brow[3],nose_tip[0]))/2
    noseB0toB3 = distance(nose_bridge[0],nose_bridge[3])
    ratiosGeneral.append(eyeB1toNoseTip4/noseB0toB3)
    ratiosGeneral.append(get_the_Ratio(nose_tip[2],chin[4],nose_tip[2],chin[8])/5)
    ratiosGeneral.append(get_the_Ratio(nose_tip[2],t_lip[3],nose_bridge[0],nose_bridge[3])/2)
    ratiosGeneral.append(get_the_Ratio(b_lip[3],b_lip[6],t_lip[3],t_lip[6])/2)
    #ratiosGeneral.append(get_the_Ratio(nose_tip[2],left_eye[0],nose_tip[2],left_eye[3])/3)
    noseT2toeye0 = (distance(nose_tip[2],left_eye[0])+distance(nose_tip[2],right_eye[3]))/2
    noseT2toeye3 = (distance(nose_tip[2], left_eye[3]) + distance(nose_tip[2], right_eye[0])) / 2
    ratiosGeneral.append(noseT2toeye0/noseT2toeye3)

    ratiosGeneral.append(get_the_Ratio(left_eye[2],left_eye[4],nose_bridge[0],chin[8]))




    if forNew == False:
        ratiosLast.append(ratiosGeneral)
    elif forNew == True:
        ratio_for_new.append(ratiosGeneral)

    if show:

        for face_landmarks in face_landmarks_list:
            pil_image = PIL.Image.fromarray(fr_image)
            d = PIL.ImageDraw.Draw(pil_image, 'RGBA')
            d.line(face_landmarks['chin'], fill=(0, 255, 0, 255), width=2)
            d.line(face_landmarks['nose_bridge'], fill=(0, 0, 0, 255), width=2)
            d.line(face_landmarks['nose_tip'], fill=(0, 0, 0, 255), width=2)
            d.line(face_landmarks['chin'][4:6], fill=(0, 0, 255, 255), width=3)

            # Make the eyebrows into a nightmare
            d.line(face_landmarks['left_eyebrow'], fill=(0, 0, 0, 255), width=2)
            d.line(face_landmarks['right_eyebrow'], fill=(0, 0, 0, 255), width=2)

            # Gloss the lips
            d.line(face_landmarks['top_lip'], fill=(0, 0, 0, 255), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(0, 0, 0, 255), width=2)

            # Chin

            # Apply some eyeliner
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(10, 0, 255, 255), width=1)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(10, 0, 250, 255), width=1)
            d.line(face_landmarks['left_eye'][2:3], fill=(0, 0, 255, 255), width=2)
            lastImage = d

    r1 = get_the_Ratio(left_eye[0],left_eye[3],right_eye[0],right_eye[3])
    r2 = get_the_Ratio(l_eye_brow[0],l_eye_brow[4],r_eye_brow[0],r_eye_brow[4])
    r3 = get_the_Ratio(t_lip[0],chin[4],t_lip[6],chin[12])



answerList = []
features = ["RATIO1", "RATIO2", "RATIO3", "RATIO4", "RATIO5", "RATIO6", "RATIO7", "RATIO8", "RATIO9", "RATIO10",
            "RATIO11", "RATIO12", "RATIO14", "RATIO15",
            "RATIO17", "RATIO18", "RATIO19", "RATIO20"]
targets = ["1", "2"]

print(len(answerList))

list_to_dataList = []
list_to_Answer = []
print(len(list_to_dataList))

# againe = True
# show = True
# while againe :
# linkForNew = input("GIVE A LINK FOT PHOTO THAT YOU WANT PLEASE: ")
#imageProcces(linkForNew, False)
#answer = str(input("PRESS 1 IF YOU THINK PERSON IS BEAUTIFUL , PRESS 2 SAY -NO COMMENT "))


"""answerList.append("1")
fl = open("dataNew.txt", "a+")
line = linkForNew

fl.writelines(line)
fl.close()
fl2 = open("answersNew.txt", "a+")
     fl2.writelines(answer)
     fl2.close()"""

















listOfLines = list()
listOfLines2 = list()
with open("dataNew.txt", "r") as myfile:
    for line in myfile:
        listOfLines.append(line.strip())

for i in listOfLines:
    list_to_dataList.append(i)

# reading answers txt and add it to da ML answers datalist
with open("answersNew.txt", "r") as myfile2:
    for line in myfile2:
        listOfLines2.append(line.strip())

for i in listOfLines2:
    answerList.append(i)
print(list_to_dataList)

for i in list_to_dataList:
    imageProcces(i, False)

dataList = ratiosLast
again = True  # for the bottom loop

model = Sequential([


    Dense(units=1, input_shape=[46]),




])
myList = answerList
myInt = 10
newList = [float(x) / myInt for x in myList]
opt1 = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
opt2 = SGD(lr=0.001)
model.compile(loss="mean_squared_error", optimizer = opt1, metrics=['accuracy'])
xs = np.array(dataList, dtype=float)
ys = np.array(newList, dtype=float)
model.fit(xs, ys, epochs=10000)
model.save("newModel311.h5")


while again:
    link = input("give the link of file")

    show = True
    ratio_for_new = []

    imageProcces(link, True)
    print(ratio_for_new)
    b = 10*model.predict([ratio_for_new])
    print(b)

    answer = input("IF IT IS TRUE PRESS -Y-,IF not press -N ,to skip press -S")

    if answer == "n":
        newPoint = "\n" + input("Give the true point for this photo: ")

        answerList.append(newPoint)
        fl = open("dataNew.txt", "a+")
        line = "\n" +link

        fl.writelines(line)
        fl.close()
        fl2 = open("answersNew.txt", "a+")
        fl2.writelines(newPoint)
        fl2.close()
