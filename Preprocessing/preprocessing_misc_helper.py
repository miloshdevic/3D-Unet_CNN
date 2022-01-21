import pydicom
import xlwt
from xlwt import Workbook
import os
import datetime

# user defined variables
########################################################################################################################
# directory containing patient directories
directory = '//mfad.mfroot.org/rchdept/radiation_oncology_research/CDeufel/Luca_Project/test_patient_data'
# name of excel file
filename = 'patient_key_autogen_'
# directory where the excel file gets saved
dest_dir = '//mfad.mfroot.org/rchdept/radiation_oncology_research/CDeufel/Luca_Project/'
########################################################################################################################

row = 0
# Workbook is created
wb = Workbook()

# Create Colors
missingstyle = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
presentstyle = xlwt.easyxf('pattern: pattern solid, fore_colour green;')

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(0, 1, 'Folder')
sheet1.write(0, 2, 'Patient ID')
sheet1.write(0, 3, 'Patient Name')
sheet1.write(0, 4, 'Date')
sheet1.write(0, 5, 'Modality')
sheet1.write(0, 6, "Applicator")
sheet1.write(0, 7, "applicator")
sheet1.write(0, 8, "Bladder")
sheet1.write(0, 9, "bladder")
sheet1.write(0, 10, "BODY")
sheet1.write(0, 11, "Normal tissue")
sheet1.write(0, 12, "Prostate")
sheet1.write(0, 13, "prostate")
sheet1.write(0, 14, "Rectum")
sheet1.write(0, 15, "rectum")
sheet1.write(0, 16, "Urethra")
sheet1.write(0, 17, "urethra")
sheet1.write(0, 18, "ptv")
sheet1.write(0, 19, "ctv")
sheet1.write(0, 0, "Pixel Spacing")


# iterate through folders in directory and read the dicom metadata from from first image
for folder in sorted(os.listdir(directory), key=str.lower):
    Applicator, Prostate, Urethra, Bladder, Rectum, BODY = "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"
    Normal_tissue, applicator, prostate, urethra, bladder, rectum, ptv, ctv = "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING",  "MISSING", "MISSING"
    Applicatorstyle, Prostatestyle, Urethrastyle, Bladderstyle, Rectumstyle, BODYstyle = missingstyle, missingstyle, missingstyle, missingstyle, missingstyle, missingstyle
    Normal_tissuestyle, applicatorstyle, prostatestyle, urethrastyle, bladderstyle, rectumstyle, ptvstyle, ctvstyle = missingstyle, missingstyle, missingstyle, missingstyle, missingstyle, missingstyle, missingstyle, missingstyle
    if folder.startswith("patient"):
        folderloc = directory + "/" + folder
        file = os.listdir(folderloc)
        fileloc = folderloc + "/" + file[1]
        df = pydicom.read_file(fileloc)
        name = str(df.PatientName)
        ID = str(df.PatientID)
        date = str(df.StudyDate)
        modality = str(df.Modality)
        try:
            spacing = str(df.PixelSpacing)
        except:
            spacing = str("unavailable")
        row = row + 1
        for eachfile in os.listdir(folderloc):
            if eachfile.startswith("RS"):
                rsfileloc = folderloc + "/" + eachfile
                rs = pydicom.read_file(rsfileloc)
                name_list = []
                for index, item in enumerate(rs.StructureSetROISequence):
                    # print(item.ROIName)
                    if str(item.ROIName).startswith("Applicator"):
                        Applicator = str(item.ROIName)
                        Applicatorstyle = presentstyle
                    elif str(item.ROIName).startswith("Prostate") or str(item.ROIName).startswith("prostate"):
                        Prostate = str(item.ROIName)
                        Prostatestyle = presentstyle
                    elif str(item.ROIName).startswith("Urethra") or str(item.ROIName).startswith("urethra"):
                        Urethra = str(item.ROIName)
                        Urethrastyle = presentstyle
                    elif str(item.ROIName).startswith("Bladder") or str(item.ROIName).startswith("bladder"):
                        Bladder = str(item.ROIName)
                        Bladderstyle = presentstyle
                    elif str(item.ROIName).startswith("Rectum") or str(item.ROIName).startswith("rectum"):
                        Rectum = str(item.ROIName)
                        Rectumstyle = presentstyle
                    elif str(item.ROIName).startswith("ptv") or str(item.ROIName).startswith("PTV"):
                        ptv = str(item.ROIName)
                        ptvstyle = presentstyle
                    elif str(item.ROIName).startswith("ctv") or str(item.ROIName).startswith("CTV"):
                        ctv = str(item.ROIName)
                        ctvstyle = presentstyle
                    name_list.append(str(str(item.ROIName) + "; "))
            else:
                continue

        # paste information into a .xls file
        print(row)
        print(ID)
        print(name)
        sheet1.write(row, 1, folder)
        sheet1.write(row, 2, ID)
        sheet1.write(row, 3, name)
        sheet1.write(row, 4, date)
        sheet1.write(row, 5, modality)
        sheet1.write(row, 6, Applicator, Applicatorstyle)
        sheet1.write(row, 8, Bladder, Bladderstyle)
        sheet1.write(row, 10, BODY, BODYstyle)
        sheet1.write(row, 11, Normal_tissue, Normal_tissuestyle)
        sheet1.write(row, 12, Prostate, Prostatestyle)
        sheet1.write(row, 14, Rectum, Rectumstyle)
        sheet1.write(row, 16, Urethra, Urethrastyle)
        sheet1.write(row, 18, ptv, ptvstyle)
        sheet1.write(row, 19, ctv, ctvstyle)
        sheet1.write(row, 20, name_list)
        sheet1.write(row, 0, spacing)

now = datetime.datetime.now()

# save .xls file
wb.save(dest_dir + filename +str(now.date())+'.xls')
