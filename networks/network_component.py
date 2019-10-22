'''Base Class for Model Components (Encoder, AUPredictor, Decoder, Discriminator)'''

import os
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn

class NetworkComponent(nn.Module):
    '''
        Base class for the networks Encoder, AUPredictor, Decoder, and Discriminator.
        Implements default methods that can be overridden in subclasses should the need arise
    '''
    def __init__(self, blocks, args, name='DefaultNetwork'):

        super(NetworkComponent, self).__init__()

        self.blocks = blocks
        self.args = args

        self.name = name

        # input and output size are number of channels for
        # convolutional layers, and other dimensions otherwise
        self.input_size = None
        self.output_size = None

        self.layers = nn.ModuleList()
        self.create_network()

    def create_network(self):
        '''
            Reads in blocks and adds modules to the ModuleList() self.layers
            if required.
            Additional block types and/or additional options for the individual
            block types can be added as necessary.
        '''

        prev_filters = 3
        out_filters = []
        for block in self.blocks:

            if block['type'] == 'net':
                prev_filters = block['channels']
                self.input_size = prev_filters
                continue

            elif block['type'] == 'convolutional':

                batch_normalise = block['batch_normalise']
                filters = block['filters']
                kernel_size = block['kernel_size']
                stride = block['stride']

                is_pad = block['pad']
                if isinstance(is_pad, int):
                    pad = is_pad
                elif is_pad:
                    pad = (kernel_size-1)/2
                else:
                    pad = 0

                activation = block['activation']

                layer_model = []
                if batch_normalise:
                    layer_model.append(nn.Conv2d(prev_filters, filters, kernel_size,
                                                 stride, pad, bias=False))
                    layer_model.append(nn.BatchNorm2d(filters, eps=1e-4))
                else:
                    layer_model.append(nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))

                if activation == 'relu':
                    layer_model.append(nn.ReLU(inplace=True))
                elif activation == 'leaky_relu':
                    layer_model.append(nn.LeakyReLU(inplace=True))
                elif activation == 'tanh':
                    layer_model.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layer_model.append(nn.Sigmoid())

                if 'initialization' in block:
                    if block['initialization'] == 'kaiming_uniform':
                        assert activation in ['relu', 'leaky_relu'], \
                            'Only use Kaiming initialization with ReLU or LeakyReLU'
                        nn.init.kaiming_uniform_(layer_model[0].weight, nonlinearity=activation)

                prev_filters = filters
                out_filters.append(prev_filters)
                self.layers.append(nn.Sequential(*layer_model))

            elif block['type'] == 'batch_norm':

                num_features = block['num_features']
                eps = block['eps']
                out_filters.append(prev_filters)

                self.layers.append(nn.BatchNorm2d(num_features=num_features, eps=eps))


            elif block['type'] == 'maxpool':

                pool_size = block['pool_size']
                stride = block['stride']
                out_filters.append(prev_filters)

                self.layers.append(nn.MaxPool2d(pool_size, stride))

            elif block['type'] == 'avgpool':

                pool_size = block['pool_size']
                stride = block['stride']
                out_filters.append(prev_filters)

                self.layers.append(nn.AvgPool2d(pool_size, stride))

            elif block['type'] == 'upsample':

                if block['output_sizes'] == 'None':
                    self.layers.append(nn.Upsample(scale_factor=block['scale_factor'],
                                                   mode=block['mode'].strip(), align_corners=False))
                else:
                    output_sizes = tuple(int(s) for s in block['output_sizes'].split(','))
                    self.layers.append(nn.Upsample(size=output_sizes,
                                                   scale_factor=block['scale_factor'],
                                                   mode=block['mode'].strip(), align_corners=False))

                out_filters.append(prev_filters)

            elif block['type'] == 'connected':

                output_features = block['output_features']
                activation = block['activation']
                use_bias = (block['use_bias'])

                layer_model = [nn.Linear(prev_filters, output_features, bias=use_bias)]
                if activation == 'linear':
                    pass
                elif activation == 'relu':
                    layer_model.append(nn.ReLU(inplace=True))
                elif activation == 'leaky_relu':
                    layer_model.append(nn.LeakyReLU(inplace=True))
                elif activation == 'sigmoid':
                    layer_model.append(nn.Sigmoid())

                if 'initialization' in block:
                    if block['initialization'] == 'kaiming_uniform':
                        assert activation in ['relu', 'leaky_relu'], \
                            'Only use Kaiming initialization with ReLU or LeakyReLU'
                        nn.init.kaiming_uniform_(layer_model[0].weight, nonlinearity=activation)

                self.layers.append(nn.Sequential(*layer_model))
                prev_filters = output_features
                out_filters.append(prev_filters)

            elif block['type'] == 'softmax':

                dim = block['dim']
                self.layers.append(nn.Softmax(dim=dim))

                out_filters.append(prev_filters)

            elif block['type'] == 'route':
                # route block splits off a branch at a block identified by
                # its 'branch_at_layer'
                current_layer_ind = len(self.layers)
                branch_at_layer = block['branch_at_layer'] if block['branch_at_layer'] >= 0 \
                                                else block['branch_at_layer'] + current_layer_ind
                prev_filters = self.input_size if branch_at_layer == -1 \
                                               else out_filters[branch_at_layer]
                out_filters.append(prev_filters)

                self.layers.append(nn.Identity())

            elif block['type'] == 'combine_concat':
                # combine_concat block combines the outputs of two previous
                # blocks by concatenation
                combine_layers = block['combine_layers'].split(',')
                current_layer_ind = len(self.layers)
                combine_layers = [int(i) if int(i) > 0 else int(i) + current_layer_ind
                                  for i in combine_layers]

                prev_filters = 0
                for ind in combine_layers:
                    if ind == -1:
                        prev_filters += self.input_size
                    else:
                        prev_filters += out_filters[ind]

                self.layers.append(nn.Identity())
                out_filters.append(prev_filters)

            elif block['type'] == 'combine_multiply':
                # combine_multiply block combines output of two previous
                # blocks by element-wise multiplication
                combine_layers = block['combine_layers'].split(',')
                current_layer_ind = len(self.layers)
                combine_layers = [int(i) if int(i) > 0 else int(i) + current_layer_ind
                                  for i in combine_layers]

                # number of filters is the number from any layer in combine_layers
                if combine_layers[0] == -1:
                    prev_filters = self.input_size
                else:
                    prev_filters = out_filters[combine_layers[0]]
                self.layers.append(nn.Identity())
                out_filters.append(prev_filters)

            elif block['type'] == 'select_subset':
                # select_subset block selects a subset of the elements of the output from
                # the given layer by selecting the indices from start_index:end_index on the
                # given dimension
                current_layer_ind = len(self.layers)
                origin_layer = block['origin_layer'] if block['origin_layer'] >= 0 \
                                                     else block['origin_layer'] + current_layer_ind
                if block['dimension'] == 1:
                    if origin_layer == -1:
                        prev_filters = self.input_size
                    else:
                        prev_filters = out_filters[origin_layer]
                else:
                    print('WARNING: Block type select_subset is unstable for dimension != 1')

                self.layers.append(nn.Identity())
                out_filters.append(prev_filters)

            elif block['type'] == 'identity':
                # Simply the identity
                self.layers.append(nn.Identity())
                out_filters.append(prev_filters)

            elif block['type'] == 'resnet50':
                # Use pretrained ResNet50 layers
                resnet50 = torchvision.models.resnet50(pretrained=True)

                end_at_layer = block['end_at_layer']
                resnet_model = nn.Sequential(*list(resnet50.children())[:end_at_layer])

                for ind, resnet_layer in enumerate(resnet_model):
                    if ind >= block['freeze_first_n_layers']:
                        break

                    for param in resnet_layer.parameters():
                        param.requires_grad = False

                self.layers.append(resnet_model)
                prev_filters = block['n_out_filters']
                out_filters.append(prev_filters)

            elif block['type'] == 'vgg-face':

                state_dict = torch.load(block['state_dict_filepath'])

                # Remove unneeded weights
                del state_dict["fc6.weight"]
                del state_dict["fc6.bias"]
                del state_dict["fc7.weight"]
                del state_dict["fc7.bias"]
                del state_dict["fc8.weight"]
                del state_dict["fc8.bias"]

                vgg_dict = OrderedDict({
                    'conv1_1': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    'relu1_1': nn.ReLU(inplace=True),
                    'conv1_2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    'relu1_2': nn.ReLU(inplace=True),
                    'pool1': nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
                    'conv2_1': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    'relu2_1': nn.ReLU(inplace=True),
                    'conv2_2': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    'relu2_2': nn.ReLU(inplace=True),
                    'pool2': nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
                    'conv3_1': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    'relu3_1': nn.ReLU(inplace=True),
                    'conv3_2': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    'relu3_2': nn.ReLU(inplace=True),
                    'conv3_3': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    'relu3_3': nn.ReLU(inplace=True),
                    'pool3': nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
                    'conv4_1': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    'relu4_1': nn.ReLU(inplace=True),
                    'conv4_2': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    'relu4_2': nn.ReLU(inplace=True),
                    'conv4_3': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    'relu4_3': nn.ReLU(inplace=True),
                    'pool4': nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
                    'conv5_1': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    'relu5_1': nn.ReLU(inplace=True),
                    'conv5_2': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    'relu5_2': nn.ReLU(inplace=True),
                    'conv5_3': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    'relu5_3': nn.ReLU(inplace=True),
                    'pool5': nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)})

                vgg_model = nn.Sequential(vgg_dict)
                vgg_model.load_state_dict(state_dict)

                for ind, vgg_layer in enumerate(vgg_model):
                    if ind >= block['freeze_first_n_layers']:
                        break

                    for param in vgg_layer.parameters():
                        param.requires_grad = False

                self.layers.append(vgg_model)
                prev_filters = 512
                out_filters.append(prev_filters)

            else:
                print('Block of type %s is not implemented! Ignoring this block' % block['type'])


        self.output_size = prev_filters

    def forward(self, batch):
        '''
            Run the network on batch.
            The method goes through the blocks and does what
        '''
        outputs = dict()
        current_result = batch
        outputs[-1] = current_result # needed if first proper block is a route/select_subset

        for index, block in enumerate(self.blocks[1:]): # skip first [net] block

            if block['type'] in ['convolutional', 'maxpool', 'avgpool', 'upsample',
                                 'batch_norm', 'identity', 'softmax', 'resnet50', 'vgg-face']:

                current_result = self.layers[index](current_result)
                outputs[index] = current_result

            elif block['type'] == 'connected':

                current_result = torch.transpose(current_result, 1, 3)

                current_result = self.layers[index](current_result)

                current_result = torch.transpose(current_result, 1, 3)
                outputs[index] = current_result

            elif block['type'] == 'route':
                # block 'route' enters a new branch, with the branching off
                # point given by 'branch_at_layer'
                branch_at_layer = block['branch_at_layer']
                if branch_at_layer < 0:
                    branch_at_layer += index

                current_result = outputs[branch_at_layer]
                outputs[index] = current_result

            elif block['type'] == 'combine_concat':
                # block 'combine_concat' combines several previous outputs
                # by concatenation along the dimension 'dimension'
                combine_layers = block['combine_layers'].split(',')
                combine_layers = [int(i) if int(i) > 0 else int(i) + index
                                  for i in combine_layers]

                current_result = torch.cat([outputs[layer] for layer in combine_layers],
                                           block['dimension'])
                outputs[index] = current_result

            elif block['type'] == 'combine_multiply':
                # combine_multiply block combines output of two previous
                # blocks by element-wise multiplication
                combine_layers = block['combine_layers'].split(',')
                combine_layers = [int(i) if int(i) > 0 else int(i) + index
                                  for i in combine_layers]

                current_result = outputs[combine_layers[0]]
                for layer in combine_layers[1:]:
                    current_result *= outputs[layer]
                outputs[index] = current_result

            elif block['type'] == 'select_subset':
                # select_subset block selects a subset of the elements of the output from
                # the given layer by selecting the indices from start_index:end_index on the
                # given dimension
                origin_layer = block['origin_layer'] if block['origin_layer'] >= 0 \
                                                     else block['origin_layer'] + index

                select_indices = torch.tensor([ind for ind in range(block['start_index'],
                                                                    block['end_index'])],
                                              dtype=torch.long, device=self.args.device)
                current_result = torch.index_select(outputs[origin_layer],
                                                    block['dimension'], select_indices)
                outputs[index] = current_result

            else:
                print('Block of type %s is not implemented! Ignoring this block' % block['type'])

        return current_result

    def load(self):
        '''
            Loads the network from load_dir/[network_name]_epoch_[load_epoch].pth
            The individual layers load the state_dict()s from the corresponding index
            in the loaded state_dict_list
        '''

        filename = '%s_epoch_%s.pth' % (self.name, self.args.load_epoch)
        filepath = os.path.join(self.args.load_dir, filename)

        assert os.path.exists(filepath), \
            'Requested %s weightsfile does not exist: %s' % (self.name, filepath)

        state_dict_list = torch.load(filepath)

        for index, _ in enumerate(self.layers):
            self.layers[index].load_state_dict(state_dict_list[index])

        print('loaded %s: %s' % (self.name, filepath))

    def save(self, n_epoch, save_path):
        '''
            Saves the weights of the network to save_path/[network_name]_epoch_[n_epoch].pth
            The state_dict()s for the individual layers are saved in a list in the correct order
        '''

        filename = '%s_epoch_%s.pth' % (self.name, n_epoch)
        filepath = os.path.join(save_path, filename)

        state_dict_list = [layer.state_dict() for layer in self.layers]
        torch.save(state_dict_list, filepath)

        print('Saved %s: %s' % (self.name, filepath))
