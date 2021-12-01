from caffe import layers as L, params as P
import caffe
from caffe.proto import caffe_pb2
# import caffe.proto
import os


def create_model(model_path, lmdb_path='train.lmdb', is_train=True):
    if not os.path.exists(lmdb_path):
        raise ("lmdb file dose not exist")
    n = caffe.NetSpec()
    if is_train:
        n.data, n.label = L.Data(
            source=lmdb_path,
            batch_size=1,
            backend=P.Data.LMDB,
            transform_param=dict(scale=1.),
            ntop=2)
    else:
        n.data = L.Input(input_param={'shape': {'dim': [1, 3, 10 ,10]}})

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=1, weight_filler=dict(type='xavier'))
    n.pooling = L.PSROIPooling(n.data, n.conv1, spatial_scale=0.0625, output_dim=3, group_size=1)


    s = n.to_proto()
    with open(model_path, 'w') as f:
        f.write(str(s))


def create_solver(solver_path, model_path):
    s = caffe_pb2.SolverParameter()
    s.train_net = model_path
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005
    # The learning rate policy
    s.lr_policy = "inv"
    s.gamma = 0.0001
    s.power = 0.75
    # Display every 100 iterations
    s.display = 1
    # The maximum number of iterations
    s.max_iter = 100000
    # snapshot intermediate results
    s.snapshot = 100000
    s.snapshot_prefix = "model.caffemodel"
    # solver mode: CPU or GPU
    # s.solver_mode = "CPU"
    s.type = "SGD"

    with open(solver_path, 'w') as f:
        f.write(str(s))


def create_caffemodel(train_proto, deploy_proto, lmdb_path, solver_proto, caffemodel_path):
    # 生成模型训练prototxt
    create_model(train_proto, lmdb_path, is_train=True)
    print('save train prototxt')
    # 生成模型部署prototxt
    create_model(deploy_proto, lmdb_path, is_train=False)
    print('save deploy prototxt')
    # 生成solver
    create_solver(solver_proto, train_proto)
    print('save solver prototxt')
    # 保存模型caffemodel
    solver = caffe.SGDSolver(solver_proto)
    solver.net.save(caffemodel_path)
    print('save caffemodel')


create_caffemodel('train.prototxt', 'psroipooling.prototxt', '../train.lmdb', 'solver.prototxt', 'psroipooling.caffemodel')
os.remove('train.prototxt')
os.remove('solver.prototxt')