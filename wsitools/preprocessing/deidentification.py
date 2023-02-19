import struct
import tifffile

# refer to:
# https://forum.image.sc/t/anonymize-svs-file/61507/2

fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs/tobe_deidentified.svs"
with tifffile.TiffFile(fn, mode='r+b') as svs:
    assert svs.is_svs
    fh = svs.filehandle
    tiff = svs.tiff
    for page in svs.pages[::-1]:
        if page.subfiletype not in (1, 9):
            break  # not a label or macro image
        # zero image data in page
        for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
            fh.seek(offset)
            fh.write(b'\0' * bytecount)
        # seek to position where offset to label/macro page is stored
        previous_page = svs.pages[page.index - 1]  # previous page
        fh.seek(previous_page.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        offset = previous_page.offset + tiff.tagnosize + tagno * tiff.tagsize
        fh.seek(offset)
        # terminate IFD chain
        fh.write(struct.pack(tiff.offsetformat, 0))
        print(f'wiped {page}')

