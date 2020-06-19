import os
import time


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = os.path.join(opt.save_dir, 'logs_{}'.format(time_str))

        print('Creating', os.path.dirname(log_dir))
        if not os.path.exists(os.path.dirname(log_dir)):
            os.mkdir(os.path.dirname(log_dir))
            print('Done')
        print('Creating', log_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            print('Done')

        self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()


