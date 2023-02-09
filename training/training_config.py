import configparser


def read_or_default(config, section, option, field_type, default=None):
    """Reads and returns a config field.

    Args:
      config(ConfigParser): ConfigParser object with loaded config data
      section(str): Section in INI file of field to be loaded
      option(str): Option in INI file of field to be loaded
      field_type(str): Field to be loaded (TODO: make actual types)
      default(field_type): Default value for field

    Returns:
      Value of field, or if not specified, default value

    """
    try:
        if(field_type == 'int'):
            return config.getint(section, option)
        elif(field_type == 'float'):
            return config.getfloat(section, option)
        elif(field_type == 'str'):
            return config.get(section, option)
        elif(field_type == 'bool'):
            return config.getboolean(section, option)
    except BaseException:
        return default


class TrainingConfig:
    """Stores data, model, and optimization settings for training.

    Attributes:
        task(str): Task to perform ('MOCO')
        num_coils(int): Number of coils included in k-space data
        data_path(str): Path to data on server
        use_gt_params(bool): Whether to provide ground truth params as inputs

        architecture: Model architecture ('CONV_RESIDUAL',or 'INTERLACER_RESIDUAL')
        kernel_size(int): Size of kernel in intermediate layers
        num_features(int): Number of features in intermediate layers
        num_convs(int): Number of convolutions per Interlacer layer
        num_layers(int): Number of layers in model
        loss(str): Loss function ('L1' or 'L2' or 'SSIM')
        input_domain(bool): Domain of network input ('IMAGE' or 'FREQ')
        input_type(str): Input data ('RAW' or 'GRAPPA' or 'NUFFT')
        output_domain(bool): Domain of network output ('IMAGE' or 'FREQ')
        enforce_dc(bool): Whether to include data consistency loss term
        hyp_model(bool): Whether to use hypernetwork based on motion params

        num_epochs(int): Number of training epochs
        batch_size(int): Batch size

    """

    def __init__(self, config_path):
        self.config_path = config_path

    def read_config(self):
        """Read in fields from INI config file."""
        config = configparser.ConfigParser()
        config.read(self.config_path)

        self.task = read_or_default(
            config, 'DATA', 'task', 'str')
        self.num_coils = read_or_default(
            config, 'DATA', 'num_coils', 'int')
        self.data_path = read_or_default(
            config, 'DATA', 'data_path', 'str')
        self.use_gt_params = read_or_default(
            config, 'DATA', 'use_gt_params', 'bool', False)

        self.architecture = read_or_default(
            config, 'MODEL', 'architecture', 'str')
        self.kernel_size = read_or_default(
            config, 'MODEL', 'kernel_size', 'int')
        self.num_features = read_or_default(
            config, 'MODEL', 'num_features', 'int')
        self.num_convs = read_or_default(
            config, 'MODEL', 'num_convs', 'int', 1)
        self.num_layers = read_or_default(
            config, 'MODEL', 'num_layers', 'int')
        self.loss = read_or_default(config, 'MODEL', 'loss', 'str')
        self.input_domain = read_or_default(
            config, 'MODEL', 'input_domain', 'str')
        self.input_type = read_or_default(
            config, 'MODEL', 'input_type', 'str')
        self.output_domain = read_or_default(
            config, 'MODEL', 'output_domain', 'str')
        self.enforce_dc = read_or_default(
            config, 'MODEL', 'enforce_dc', 'bool')
        self.hyp_model = read_or_default(
            config, 'MODEL', 'hyp_model', 'bool', False)
        self.motinp_model = read_or_default(
            config, 'MODEL', 'motinp_model', 'bool', False)

        self.num_epochs = read_or_default(
            config, 'TRAINING', 'num_epochs', 'int')
        self.batch_size = read_or_default(
            config, 'TRAINING', 'batch_size', 'int')
        self.set_job_name()

    def set_job_name(self):
        """Set job name for storing training logs."""
        self.job_name = ''
        for tag in [
                self.task,
                self.num_coils,
                self.use_gt_params,
                self.architecture,
                self.kernel_size,
                self.num_features,
                self.num_convs,
                self.num_layers,
                self.loss,
                self.input_domain,
                self.input_type,
                self.output_domain,
                self.hyp_model,
                self.motinp_model,
                self.enforce_dc,
                self.num_epochs,
                self.batch_size]:
            if(tag is not None):
                self.job_name += '-' + str(tag)
        self.job_name += ''
        if(self.job_name[0]=='-'):
            self.job_name = self.job_name[1:]
