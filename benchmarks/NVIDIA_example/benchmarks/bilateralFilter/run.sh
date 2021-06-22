make clean ; make ;
cp data/1024nature.bmp data/nature_monte.bmp;
./bilateralFilter -radius=2 -passes=1 -file=ref.ppm -benchmark ;
cp data/ref_nature_monte.bmp data/nature_monte.bmp;
./bilateralFilter -radius=2 -passes=1 -file=ref.ppm 
