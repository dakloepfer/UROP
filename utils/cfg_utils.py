''' Contains helper functions that have loosely to do with the .cfg file '''

def parse_cfg(cfgfile):
    '''
        Takes in a filepath to a .cfg file and returns a dictionary with one entry
        for each network component (encoder, decoder, ...).
        Each component is a list of dictionaries, each dictionary describing
        the options for one block.

        Each new network component starts with the name in curly brackets:
        {networkname} \n
        [blocks]
        ...
        Each new block starts with the type in square brackets:
        [type] \n
        ...other options...

        Note: Might need to add another layer, if eg the encoder should have some sub-networks
    '''
    cfg = dict()
    current_network_component = None

    blocks = []
    block = None

    config_file = open(cfgfile, 'r')

    print('Network cfg file:')
    line = config_file.readline()
    while line != '':
        print(line)
        line = line.strip()

        if line == '' or line[0] == '#':
            line = config_file.readline()
            continue

        elif line[0] == '{':
            if current_network_component:
                blocks.append(block)
                cfg[current_network_component] = blocks
                blocks = []
                block = None

            current_network_component = line.lstrip('{').rstrip('}')

        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')

        else:
            key, value = line.split('=')
            key = key.strip()

            if key == 'type':
                key = '_type'

            value = value.strip()
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value

            block[key] = value

        line = config_file.readline()

    if block:
        blocks.append(block)
        cfg[current_network_component] = blocks

    config_file.close()

    return cfg
