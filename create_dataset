import logging

def Create_dataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
   
        img_data_array=[]
        class_name=[]
        n_classes = 0
        
        for dir1 in os.listdir(selfdata_path):
            for file in os.listdir(os.path.join(self.data_path, dir1)):
                if file not f.endswith('.DS_store')]
                image_path= os.path.join(self.data_path, dir1,  file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                try:
                    #zoom of the image with original proportion
                    image = cv2.resize(image, None, fx = 2, fy = 2, interpolation=cv2.INTER_LINEAR)
            
                except:
                    log = logging.error("Exception occurred", exc_info=True)
                    print('Error resize ' + file)
                    print(log)
                #img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                #img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                image = np.array(image)
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image = image.astype('float32')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.medianBlur(image,5)
                
                image /= 255 
                img_data_array.append(image)
                class_name.append(dir1)
                n_classes += 1
        return img_data_array, np.array(class_name),n_classes