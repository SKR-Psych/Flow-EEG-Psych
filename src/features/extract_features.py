from sklearn.base import BaseEstimator, TransformerMixin
import mne

class FBCSPExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_bands, n_components=3):
        self.frequency_bands = frequency_bands
        self.n_components = n_components
        self.csp_pipelines = {}
    
    def fit(self, X, y):
        for band, (fmin, fmax) in self.frequency_bands.items():
            X_filtered = mne.filter.filter_data(X, sfreq=256, l_freq=fmin, h_freq=fmax, verbose=False)
            csp = mne.decoding.CSP(n_components=self.n_components)
            csp.fit(X_filtered, y)
            self.csp_pipelines[band] = csp
        return self
    
    def transform(self, X):
        features = []
        for band, csp in self.csp_pipelines.items():
            X_filtered = mne.filter.filter_data(X, sfreq=256, l_freq=self.frequency_bands[band][0], 
                                               h_freq=self.frequency_bands[band][1], verbose=False)
            features.append(csp.transform(X_filtered))
        return np.hstack(features)





