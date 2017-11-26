#! /usr/bin/env python

import argparse
import os
import sys
import getpass
from numpy import genfromtxt
import pyxnat


def getxnatuploaddetails(row):
    def validatestring(input):
        output = input.translate(None, '!@#$=')
        return output

    project = row[0]
    subject = row[1]
    experiment = row[2]
    scan = row[3]
    filepath = os.path.abspath(row[4])
    valid = True
    if (len(project) < 1) or (len(subject) < 1) or (len(experiment) < 1) or (len(scan) < 1) or (len(filepath) < 1):
        print 'invalid information', line
        valid = False
    if os.path.exists(filepath) is False:
        print 'file not found', filepath
        valid = False
    return valid, validatestring(project), validatestring(subject), validatestring(experiment), validatestring(
        scan), filepath


def getxnatspreadsheetfilepattern():
    pattern = [
        r'project',
        r'subject-id',
        r'session-label',
        r'scan-label',
        r'image-file-path',
    ]

    dtype = ''
    dtype += 'a50,'  # filename
    dtype += 'a50,'  # subject-id
    dtype += 'a50,'  # session-label
    dtype += 'a50,'  # scan-label
    dtype += 'a500,'  # image filepath

    delimiter = ' '
    numberoflinesinheadre = 0

    return pattern, dtype, delimiter, numberoflinesinheadre


parser = argparse.ArgumentParser(description='XNAT usage example')
parser.add_argument('-i', '--server',
                    dest='server',
                    metavar='server',
                    help='XNAT server from where the data is taken',
                    required=True,
                    default='https://cmic-xnat.cs.ucl.ac.uk')
parser.add_argument('-u', '--username',
                    dest='username',
                    metavar='username',
                    help='xnat server username',
                    required=True,
                    default='ntoussaint')
parser.add_argument('-p', '--project',
                    dest='project',
                    metavar='project',
                    help='xnat server project',
                    required=False,
                    nargs='+',
                    default='ADNI')
parser.add_argument('-s', '--subject',
                    dest='subject',
                    metavar='subject',
                    help='xnat server subject',
                    required=False,
                    nargs='+')
parser.add_argument('-e', '--experiment',
                    dest='experiment',
                    metavar='experiment',
                    nargs='+',
                    help='xnat server experiment',
                    required=False)
parser.add_argument('-a', '--scan',
                    dest='scan',
                    metavar='scan',
                    nargs='+',
                    help='xnat server scan',
                    required=False)
parser.add_argument('-f', '--filepath',
                    dest='filepath',
                    metavar='filepath',
                    nargs='+',
                    help='file to upload',
                    required=False)
parser.add_argument('-c', '--spreadsheet',
                    dest='spreadsheet',
                    metavar='spreadsheet',
                    help='csv file with information to upload, the template is <project> <subject-id> <session-label> <scan-label> <image-file-path>',
                    required=False)

args = parser.parse_args()

current_dir = os.getcwd()

if (args.server is None) or (args.username is None):
    print 'ERROR: Please provide a server and username'
    sys.exit()
if (args.project is None) or (args.subject is None) or (args.experiment is None) or (args.scan is None) or (
            args.filepath is None):
    if args.spreadsheet is None:
        print 'ERROR: Please provide a spreadsheet OR a list of information'
        sys.exit()
else:
    if args.spreadsheet is not None:
        print 'ERROR: Please provide a spreadsheet OR a list of information'
        sys.exit()

pwd = getpass.getpass()

xnat = pyxnat.Interface(args.server, args.username, pwd, '/tmp/')

projects = []
subjects = []
experiments = []
scans = []
filepaths = []

if args.spreadsheet is not None:
    pattern, dtype, delimiter, numberoflinesinheader = getxnatspreadsheetfilepattern()
    print 'the required pattern is the following'
    print 'pattern: ', pattern, ' -- delimiter: ', delimiter, ' -- header no of lines: ', numberoflinesinheader
    datamatrix = genfromtxt(args.spreadsheet, delimiter=delimiter, autostrip=True, dtype=dtype,
                            skip_header=numberoflinesinheader)
    print 'The spreadsheet contains ', datamatrix.size, ' files to upload\n\n'
    for line in datamatrix:
        valid, project, subject, experiment, scan, filepath = getxnatuploaddetails(line)
        if valid == False:
            continue
        projects.append(project)
        subjects.append(subject)
        experiments.append(experiment)
        scans.append(scan)
        filepaths.append(filepath)

else:
    projects = args.project
    subjects = args.subject
    experiments = args.experiment
    scans = args.scan
    filepaths = args.filepath

for i in range(len(projects)):
    print 'uploading file for information', projects[i], subjects[i], experiments[i], scans[i], filepaths[i]
    m_project = xnat.select.project(projects[i])
    m_subject = m_project.subject(subjects[i])
    m_experiment = m_subject.experiment(experiments[i])
    m_scan = m_experiment.scan(scans[i])
    if not m_scan.exists():
        m_scan.insert(scans='xnat:mrScanData')
    filebasename = os.path.basename(filepaths[i])
    m_file = m_scan.resource('NIFTI').file(filebasename)
    if not m_file.exists():
        try:
            print 'Uploading file: \t', filepaths[i]
            retvalue = m_file.insert(filepaths[i])
            print "Done..."
        except:
            print "there was an error uploading file... "
            e = sys.exc_info()
            print e[0], e[1], e[2]
    else:
        print 'file already exist: \t', filepaths[i]


