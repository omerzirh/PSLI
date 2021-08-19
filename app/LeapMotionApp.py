import Leap, sys, time,csv 
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import numpy as np  
import Tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

images = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
class LeapMotionListener(Leap.Listener):
    
    def on_init(self,controller):
        print "Loaded."
    
    def on_connect(self,controller):
        self.i=0
        print "Leap Motion Connected."
        
    def on_disconnect(self,controller):
        print "Leap Motion disconnected."
    
    def on_frame(self,controller):
        self.i+=1
        frame=controller.frame()

        for i,hand in enumerate(frame.hands):
            typeHand=0 if hand.is_left else 1
        
            if(typeHand==0 and i==0):#left 1. Hand  
                pass 

            elif (typeHand==1 and i==0): #right 1 hand  
                dataSample =getAngles(frame)
                dataToSave = dataSample
                predicted = model.predict([dataSample])
                globalList= globals()                
                signIndex = str(images[(globalList['step'])])
                if(str(predicted)=='['+signIndex+']'):
                    matchedArray= np.append(dataToSave,signIndex)
                    writeToCsv(matchedArray,'collectedData.csv')
                    ApprovalLabel(globalList['f2'],"Affirmative") 
                    onClickNextButton((globalList['step']),globalList['f2'])
  
                writeToCsv(dataToSave,'data.csv')
                time.sleep(1.6)        
     
def get_cos_angle_2vec(v1, v2):
    # return the cos(angle) of vectors
    return np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)

def getAngles(frame):
    
    anglePoints = []
    # palm 
    P0 = np.array(frame.hands[0].palm_position.to_float_array())
    # thumb
    P1 = np.array(frame.hands[0].fingers[0].bone(1).next_joint.to_float_array())
    P2 = np.array(frame.hands[0].fingers[0].bone(2).next_joint.to_float_array())
    P3 = np.array(frame.hands[0].fingers[0].bone(3).next_joint.to_float_array())
    # index
    P4 = np.array(frame.hands[0].fingers[1].bone(0).next_joint.to_float_array())
    P5 = np.array(frame.hands[0].fingers[1].bone(1).next_joint.to_float_array())
    P6 = np.array(frame.hands[0].fingers[1].bone(2).next_joint.to_float_array())
    P7 = np.array(frame.hands[0].fingers[1].bone(3).next_joint.to_float_array())
    # middle
    P8  = np.array(frame.hands[0].fingers[2].bone(0).next_joint.to_float_array())
    P9  = np.array(frame.hands[0].fingers[2].bone(1).next_joint.to_float_array())
    P10 = np.array(frame.hands[0].fingers[2].bone(2).next_joint.to_float_array())
    P11 = np.array(frame.hands[0].fingers[2].bone(3).next_joint.to_float_array())
    # ring    
    P12 = np.array(frame.hands[0].fingers[3].bone(0).next_joint.to_float_array())
    P13 = np.array(frame.hands[0].fingers[3].bone(1).next_joint.to_float_array())
    P14 = np.array(frame.hands[0].fingers[3].bone(2).next_joint.to_float_array())
    P15 = np.array(frame.hands[0].fingers[3].bone(3).next_joint.to_float_array())
    # pinky    
    P16 = np.array(frame.hands[0].fingers[4].bone(0).next_joint.to_float_array())
    P17 = np.array(frame.hands[0].fingers[4].bone(1).next_joint.to_float_array())
    P18 = np.array(frame.hands[0].fingers[4].bone(2).next_joint.to_float_array())
    P19 = np.array(frame.hands[0].fingers[4].bone(3).next_joint.to_float_array())
    
    # A1-4
    A = get_cos_angle_2vec((P1-P0), (P4-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P4-P0), (P8-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P8-P0), (P12-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P12-P0), (P16-P0))
    anglePoints.append(A)
    
    # thumb A5,6 
    A = get_cos_angle_2vec((P2-P1), (P1-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P3-P2), (P2-P1))
    anglePoints.append(A)
    
    #  Index A7-9 
    A = get_cos_angle_2vec((P5-P4), (P4-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P6-P5), (P5-P4))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P7-P6), (P6-P5))
    anglePoints.append(A)
    
    # middle A10-12
    A = get_cos_angle_2vec((P9-P8), (P8-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P10-P9), (P9-P8))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P11-P10), (P10-P9))
    anglePoints.append(A)
    
    # ring A13-15
    A = get_cos_angle_2vec((P13-P12), (P12-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P14-P13), (P13-P12))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P15-P14), (P14-P13))
    anglePoints.append(A)
    
    # pinky A16-18 
    A = get_cos_angle_2vec((P17-P16), (P16-P0))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P18-P17), (P17-P16))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P19-P18), (P18-P17))
    anglePoints.append(A)
    
    # last A19-22
    A = get_cos_angle_2vec((P3-P2), (P7-P6))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P7-P6), (P11-P10))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P11-P10), (P15-P14))
    anglePoints.append(A)
    A = get_cos_angle_2vec((P15-P14), (P19-P18))
    anglePoints.append(A)    
    
    return np.array(anglePoints)

def writeToCsv(fdataToSave,filename):
    writer = csv.writer(open('dataset/'+filename,'a'))
    writer.writerow(fdataToSave)
def ApprovalLabel(frame,labelText):
    
    label2 = tk.Label(frame, text=labelText)
    label2.grid(column=3, row=0, pady = 2)

def raise_frame(frame):
    frame.tkraise()
def ImageUpdate(signPath,frame):

    sign = Image.open("signs/"+str(images[signPath])+".png")
    sign = ImageTk.PhotoImage(sign)
    sign_label = tk.Label(frame,image= sign)
    sign_label.image = sign
    sign_label.grid(column=6,row=0)
def onClickNextButton(step,frame):
    listOfGlobals = globals()
    if(int(listOfGlobals['step'])<20):
        listOfGlobals['step']+=1
        ApprovalLabel(frame," ")
        ImageUpdate(listOfGlobals['step'],frame)
def onClickPrevButton(step,frame):
    listOfGlobals = globals()
    if(int(listOfGlobals['step'])>0):
        listOfGlobals['step']-=1
        ImageUpdate(listOfGlobals['step'],frame)

def guiSign():
    print "gui"
    root = tk.Tk()
    f1 = tk.Frame(root)
    f2 = tk.Frame(root)
    f3 = tk.Frame(root)
    f4 = tk.Frame(root)
    root.geometry('600x500')
    for frame in (f1, f2, f3, f4):
        frame.grid(row=3, column=6,columnspan=6, rowspan=3, sticky='news')
    consentText = "We want to collect your data to increase our application's accuracy. We will only get the data of your hand from Leap Motion Controller and predicted sign. No personal information included. Do you allow?"
    button1=tk.Button(f1, text='Yes', command=lambda:raise_frame(f2))
    label1=tk.Message(f1, text=consentText)
    label1.grid(row=0,column=0,sticky='news')
    button1.grid(row=1,column=0,sticky='news')

    buttonGt3=tk.Button(f1, text='No', command=lambda:raise_frame(f3))
    buttonGt3.grid(column=1, row=1, pady = 2)

 
    global step
    ImageUpdate(step,f2)
    listOfGlobal = globals()
    listOfGlobal['f2'] = f2
    print step
    label2 = tk.Label(f2, text='')
    label2.grid(column=1, row=0, pady = 2)
    buttonNxt = tk.Button(f2, text='Next', command=lambda:onClickNextButton(step,f2))
    buttonNxt.grid(column=3,row=1,pady=2)
    buttonPrv = tk.Button(f2, text='Previous', command=lambda:onClickPrevButton(step,f2))
    buttonPrv.grid(column=2,row=1,pady=2)
    buttonGt3=tk.Button(f2, text='Go to MainPage', command=lambda:raise_frame(f1))
    buttonGt3.grid(column=1, row=1, pady = 2)

    label3= tk.Label(f3, text='')
    button3= tk.Button(f3, text='Go to MainPage', command=lambda:raise_frame(f1))
    button3.grid(row=3,column=2,sticky='news')
    label3.grid(row=3,column=1,sticky='news')

    label4= tk.Label(f4, text='FRAME 4')
    button4=tk.Button(f4, text='Goto to MainPage', command=lambda:raise_frame(f1))
    button4.grid(row=3,column=2,sticky='news')
    label4.grid(row=3,column=1,sticky='news')

    raise_frame(f1)
    root.mainloop()
        

def main():
    global model
    global step
    global f2
    step=0
    xx = pd.read_csv('dataset/angles.csv')
    yy = pd.read_csv('dataset/labels.csv')
    x=xx
    y = yy['sign_id']
    model = RandomForestClassifier(random_state=42)     #njobs -1            
    model.fit(x,y)                
    
    listener=LeapMotionListener()
    controller=Leap.Controller()
    
    controller.add_listener(listener)
    
    print "Press enter to quit"
    guiSign()
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)

if __name__== "__main__":
    main()

