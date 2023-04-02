
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.model == 'poselstm':
        from .poselstm_model import PoseLSTModel
        model = PoseLSTModel()
    elif opt.model == 'onlyStudent':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


# For KD Teacher Model
def KD_create_Tmodel(opt):
    model = None
    print(opt.T_model)
    if opt.T_model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.T_model == 'poselstm':
        from .poselstm_model import PoseLSTModel
        model = PoseLSTModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.T_model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

# For KD Student Model
def KD_create_Smodel(opt):
    model = None

    from .student_model import studentModel
    model = studentModel()

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model