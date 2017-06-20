import csv
import re
from os import listdir
from os.path import isfile


def main():
  writeParagraph = False
  writeEntireText = True

  # Create a list of the relatives paths to all of the raw files.
  samples = []
  for f in listdir('raw_data/'):
    if f.endswith('.txt') and isfile('raw_data/%s' % f):
      samples.append(f)

  for sample in samples:
    filename = sample.replace('.txt', '')
    author = filename.partition('_')[0]
    header = 'Produced by'
    footer = 'End of'

    f = open('raw_data/%s.txt' % filename, 'r')
    data = f.read()
    f.close()

    # Get everything after the header
    # data = data.partition(header)[2]

    # Get everything before the footer
    # data = data.partition(footer)[0]

    # Create a list out of the lines in the file.
    data = data.split('\n')

    # Remove who produced the project gutenberg work.
    # del data[0]

    # Remove everything from each of the lines except for spaces and words.
    # data = [KeepWordsSpaces(line) for line in data]

    # Remove any leading whitespace characters and spaces from each line.
    data = [line.lstrip() for line in data]
    # Remove any trailing whitespace characters and spaces from each line.
    data = [line.rstrip() for line in data]

    # Write out each line to a csv file with column 1 being the author name
    # and column 2 being the sentence or paragraph.
    with open('csv_data/%s.csv' % filename, 'w') as csvfile:
      writer = csv.writer(csvfile)
      if writeParagraph:
        # Add each line of the paragraph to a string separate by spaces.
        paragraph = ''
        for line in data:
          if line:
            paragraph = paragraph + line + ' '
          else:
            if paragraph:
              # Remove the final space on end of each paragraph.
              paragraph = paragraph.rstrip()
              writer.writerow([author, paragraph])
            paragraph = ''
      elif writeEntireText:
        # Create csv file with single line contain entire text.
        entireText = ''
        for line in data:
          if line:
            entireText = entireText + line + ' '
        entireText.rstrip()
        writer.writerow([author, entireText])
      else:
        # Create csv file with each line as a new row.
        for line in data:
          # If the line is not an empty string.
          if line:
            writer.writerow([author, line])

if __name__ == '__main__':
  main()


