from flask import Flask, request
import pickle, json, cv2, math, threading,os,glob,requests
#from test_model import 
import predict

app = Flask(__name__)

# Endpoint to receive image data
@app.route('/', methods=['POST'])
def receiveImage():
    global target,label
    #clear the image in the save directory
    imagedir = os.path.join(os.getcwd(),'predict_result/result')
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    filelist = glob.glob(os.path.join(imagedir, "*.jpg"))
    for f in filelist:
        os.remove(f)

    content = request.data
    frame = pickle.loads(content)
    #save received photo
    cv2.imwrite("./predict_result/result/target.jpg", frame)  

    target,result_ID = predict.test()
    print('\n==================\nPredicted result: '+target\
        +"\n==================\nTarget ID: "+str(result_ID)\
            +'\n==================')
    response = make_response("{}:{}".format(result_ID, target))
    response.mimetype="text/plain"
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8123)
