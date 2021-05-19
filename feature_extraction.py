class FeatureExtractor(object):

    def __init__(self, feature_extractor, model, out_dim=20, scale=None,
                 subsample=100):

        # the provided feature extractor
        self.feature_extractor = feature_extractor

        # the clustering model
        self.model = model

        # the scaler -> scale feats to the same interval
        self.scale = scale

        # if defined, we collect subsample SIFT desc form each img
        self.subsample = subsample

    def get_descriptor(self, img_path):
        # read image
        img = cv2.imread(img_path)
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get kp and descriptors, we use descriptors only
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs


    def fit_model(self, data_list):
        training_feats = []
        # we extact SIFT descriptors
        for img_path in tqdm(data_list, desc='Fit extraction'):
            # we get sift desctiptors for img_path
            descs = self.get_descriptor(img_path)

            # if None == no descriptors we continue
            if descs is None:
                continue
            
            # if subsample, we choose randomly subsample descriptors
            if self.subsample:
                sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                descs = descs[sub_idx, :]

            # we append descriptors
            training_feats.append(descs)
        # we concatenate
        training_feats = np.concatenate(training_feats)
        print('--> Model trained on {} features'.format(training_feats.shape))
        # we fit the model
        self.model.fit(training_feats)
        print('--> Model fitted')


    def fit_scaler(self, data_list):
        # similar to fit_model
        # we get features mapped from the model
        features = self.extract_features(data_list)
        print('--> Scale trained on {}'.format(features.shape))
        # we fit the scaler
        self.scale.fit(features)
        print('--> Scale fitted')


    def extract_features(self, data_list):
        # we init features
        features = np.zeros((len(data_list), self.model.n_clusters))

        for i, img_path in enumerate(tqdm(data_list, desc='Extraction')):
            # get descriptor
            descs = self.get_descriptor(img_path)
            # 2220x128 descs
            preds = self.model.predict(descs)
            # 2220x1
            histo, _ = np.histogram(preds, bins=np.arange(self.model.n_clusters+1), density=True)
            # append histogram
            features[i, :] = histo

        return features


    def scale_features(self, features):
        # we return the normalized features
        return self.scale.transform(features)