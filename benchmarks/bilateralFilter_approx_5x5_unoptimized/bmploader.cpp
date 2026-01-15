#include <stdio.h>
#include <stdlib.h>

#pragma pack(push, 1)
typedef struct {
  short type;
  int size;
  short reserved1;
  short reserved2;
  int offset;
} BMPHeader;

typedef struct {
  int size;
  int width;
  int height;
  short planes;
  short bitsPerPixel;
  unsigned compression;
  unsigned imageSize;
  int xPelsPerMeter;
  int yPelsPerMeter;
  int clrUsed;
  int clrImportant;
} BMPInfoHeader;
#pragma pack(pop)

extern "C" void LoadBMPFile(unsigned int **dst, unsigned int *width,
                            unsigned int *height, const char *name) {
  BMPHeader hdr;
  BMPInfoHeader infoHdr;
  FILE *fd = fopen(name, "rb");
  if (!fd) {
    fprintf(stderr, "BMP load error: cannot open '%s'\n", name);
    exit(EXIT_FAILURE);
  }

  if (fread(&hdr, sizeof(hdr), 1, fd) != 1 || hdr.type != 0x4D42) {
    fprintf(stderr, "BMP load error: bad file format\n");
    exit(EXIT_FAILURE);
  }
  if (fread(&infoHdr, sizeof(infoHdr), 1, fd) != 1) {
    fprintf(stderr, "BMP load error: truncated header\n");
    exit(EXIT_FAILURE);
  }
  if (infoHdr.bitsPerPixel != 24 || infoHdr.compression) {
    fprintf(stderr, "BMP load error: need uncompressed 24-bit BMP\n");
    exit(EXIT_FAILURE);
  }

  *width = (unsigned)infoHdr.width;
  *height = (unsigned)infoHdr.height;
  unsigned n = (*width) * (*height);
  *dst = (unsigned int *)malloc(n * sizeof(unsigned int));
  if (!*dst) {
    fprintf(stderr, "BMP load error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  fseek(fd, hdr.offset - (long)sizeof(hdr) - (long)sizeof(infoHdr), SEEK_CUR);

  for (int y = 0; y < infoHdr.height; y++) {
    for (int x = 0; x < infoHdr.width; x++) {
      unsigned b = (unsigned char)fgetc(fd);
      unsigned g = (unsigned char)fgetc(fd);
      unsigned r = (unsigned char)fgetc(fd);
      (*dst)[y * infoHdr.width + x] = r | (g << 8) | (b << 16);
    }
    for (int x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++) {
      fgetc(fd);
    }
  }

  if (ferror(fd)) {
    fprintf(stderr, "BMP load error while reading pixels\n");
    free(*dst);
    exit(EXIT_FAILURE);
  }
  fclose(fd);
}
