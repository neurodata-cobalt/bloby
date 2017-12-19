import sys
import subprocess

SCRIPT_SRC = './bloby.py'
SCRIPT_PATH = './bin/bloby'

def _get_python_path():
	result = subprocess.run(['which', 'python3'], stdout=subprocess.PIPE)
	return result.stdout.decode('utf-8').replace('\n', '')

def _install_requirements():
	print('Installing Bloby requirements...')
	subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

def _create_bin_script():
	print('Creating executable...')
	src = '#!{}\n'.format(_get_python_path())
	with open(SCRIPT_SRC, 'r') as src_file:
		src += src_file.read()

	with open(SCRIPT_PATH, 'w') as out_file:
		out_file.write(src)

def _chmod_ex_script():
	print('Setting up permissions...')
	subprocess.run(['chmod', '+x', SCRIPT_PATH])

def install():
	_install_requirements()
	_create_bin_script()
	_chmod_ex_script()

	from colorama import init as color_init
	from colorama import Fore
	color_init()

	print(Fore.GREEN + 'Bloby setup successful! Run {} to start the Bloby CLI'.format(SCRIPT_PATH))

def clean():
	subprocess.run(['rm', './bin/bloby'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

if __name__ == '__main__':
	task = sys.argv[1]

	if task == 'install':
		install()
	elif task == 'clean':
		clean()
	else:
		print(Fore.RED + '`{}` not supported'.format(task))
