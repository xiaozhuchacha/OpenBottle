import sys


if __name__ == '__main__':
	fin = open(sys.argv[1], 'r')
	fout = open('./annotation_plugin_grammar.txt', 'w')

	annotation_num = int(fin.readline().rstrip('\n'))
	print annotation_num
	annotation_count = 0

	while annotation_count < annotation_num:
		fin.readline();
		annotation_count += 1


	fin.readline();

	sample_num = int(fin.readline().rstrip('\n'))
	print sample_num
	sample_count = 0;

	while sample_count < sample_num:
		sample = fin.readline().rstrip('\n')

		left_parts = []
		left_parts = sample.split(' ] ')

		for part in left_parts:
			left_bracket = part.find('[')

			annotation = part[0 : left_bracket].rstrip(' ')

			if annotation == '0':
				fout.write('approach ')
			elif annotation == '1':
				fout.write('move ')
			elif annotation == '2':
				fout.write('grasp_left ')
			elif annotation == '3':
				fout.write('grasp_right ')
			elif annotation == '4':
				fout.write('ungrasp_left ')
			elif annotation == '5':
				fout.write('ungrasp_right ')
			elif annotation == '6':
				fout.write('twist ')
			elif annotation == '7':
				fout.write('push ')
			elif annotation == '8':
				fout.write('neutral ')
			elif annotation == '9':
				fout.write('pull ')
			elif annotation == '10':
				fout.write('pinch ')
			elif annotation == '11':
				fout.write('unpinch ')

		fout.write('\n\n');

		sample_count += 1

	fin.close()
	fout.close()