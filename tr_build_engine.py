import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

## Settings
FROZEN_FPATH = 'data/frozen.pb' # ADJUST
ENGINE_FPATH = 'data/engine.plan' # ADJUST
INPUTE_NODE = 'net/input' # ADJUST
ONPUT_NODE = 'net/fc8/BiasAdd' # ADJUST
INPUT_SIZE = [3, 224, 224] # ADJUST
MAX_BATCH_SIZE = 1 # ADJUST
MAX_WORKSPACE = 1 << 20 # ADJUST

## Convert TF Frozen graph to UFF graph
uff_model = uff.from_tensorflow_frozen_model(FROZEN_FPATH, [ONPUT_NODE])

## Create UFF parser and logger
parser = uffparser.create_uff_parser()
parser.register_input(INPUTE_NODE, INPUT_SIZE, 0)
parser.register_output(ONPUT_NODE)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

## Build Optimized inference engine
engine = trt.utils.uff_to_trt_engine(
    G_LOGGER, uff_model, parser, MAX_BATCH_SIZE, MAX_WORKSPACE)

## Save inference engine
trt.utils.write_engine_to_file(ENGINE_FPATH, engine.serialize())

## Clean UP
parser.destroy()
engine.destroy()


