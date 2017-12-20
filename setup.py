import sys
import os
import subprocess

def _install_requirements():
	print('Installing Bloby requirements...')
	subprocess.run(['pip3', 'install', '-r', 'requirements.txt'])

def _create_intern_config():
	print('Creating intern.cfg...')
	boss_token = os.environ['BOSS_TOKEN']
	with open('intern_cfg_tmpl', 'r') as cfg_file:
		cfg_str = cfg_file.read()

	with open('intern.cfg', 'w') as cfg_file:
		cfg_file.write(cfg_str.format(boss_token))

def install():
	_install_requirements()
	_create_intern_config()

	from colorama import init as color_init
	from colorama import Fore
	color_init()

	print(Fore.GREEN + 'Bloby setup successful! Run {} to start the Bloby CLI'.format(SCRIPT_PATH))

if __name__ == '__main__':
	task = sys.argv[1]

	if task == 'install':
		install()
	elif task == 'clean':
		clean()
	else:
		print(Fore.RED + '`{}` not supported'.format(task))
