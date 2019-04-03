import warnings
from collections import defaultdict
import numpy
from keras.callbacks import Callback


class ModelCheckpointOnBatch(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointOnBatch, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.batches_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = numpy.less
            self.best = numpy.Inf
        elif mode == 'max':
            self.monitor_op = numpy.greater
            self.best = -numpy.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = numpy.greater
                self.best = -numpy.Inf
            else:
                self.monitor_op = numpy.less
                self.best = numpy.Inf

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            p = int(round(((batch + 1) / self.period)))
            filepath = self.filepath.format(period=p, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nBatch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (batch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nBatch %05d: %s did not improve from %0.5f' %
                                  (batch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nBatch %05d: saving model to %s' % (batch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class LossEarlyStopping(Callback):
    def __init__(self, metric, value, mode="less"):
        super(LossEarlyStopping, self).__init__()
        self.metric = metric
        self.value = value
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        if self.mode == "less":
            if logs[self.metric] < self.value:
                self.model.stop_training = True
                print('Early stopping - {} is {} than {}'.format(self.metric,
                                                                 self.mode,
                                                                 self.value))
        if self.mode == "more":
            if logs[self.metric] > self.value:
                self.model.stop_training = True
                print('Early stopping - {} is {} than {}'.format(self.metric,
                                                                 self.mode,
                                                                 self.value))


class WeightsCallback(Callback):
    def __init__(self, parameters=None, stats=None, merge_weights=True):
        super(WeightsCallback, self).__init__()
        self.layers_stats = defaultdict(dict)
        self.fig = None
        self.parameters = parameters
        self.stats = stats
        self.merge_weights = merge_weights
        if parameters is None:
            self.parameters = ["W"]
        if stats is None:
            self.stats = ["mean", "std"]

    def get_trainable_layers(self):
        layers = []
        for layer in self.model.layers:
            if "merge" in layer.name:
                for l in layer.layers:
                    if hasattr(l, 'trainable') and l.trainable and len(
                            l.weights):
                        if not any(x.name == l.name for x in layers):
                            layers.append(l)
            else:
                if hasattr(layer, 'trainable') and layer.trainable and len(
                        layer.weights):
                    layers.append(layer)
        return layers

    def on_train_begin(self, logs={}):
        for layer in self.get_trainable_layers():
            for param in self.parameters:
                if any(w for w in layer.weights if param in w.name.split("_")):
                    name = layer.name + "_" + param
                    self.layers_stats[name]["values"] = numpy.asarray(
                        []).ravel()
                    for s in self.stats:
                        self.layers_stats[name][s] = []