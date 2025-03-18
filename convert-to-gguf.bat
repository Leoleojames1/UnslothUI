@echo on
:: This batch file is used to convert Hugging Face models to GGUF format using llama.cpp
:: Usage: 
::   convert_to_gguf.bat [output_dir] [model_name] [quantization_type]
::
:: Parameters:
::   output_dir - Directory where converted files will be saved
::   model_name - Name of the model to convert
::   quantization_type - Type of quantization (q8_0, f16, etc.)
::
:: Example:
::   convert_to_gguf.bat D:\models\output my-model q8_0

:: Go to the specified directory
cd %1

:: Create converted directory if it doesn't exist
if not exist "%1\converted" mkdir "%1\converted"

:: Run the conversion script
python llama.cpp\convert-hf-to-gguf.py --outtype %3 --model-name %2-%3 --outfile %1\converted\%2-%3.gguf %2

:: The following lines are commented out but can be uncommented to generate additional quantization formats
@REM python llama.cpp\convert-hf-to-gguf.py --outtype f16 --model-name %2-f16 --outfile %1\converted\%2-f16.gguf %2
@REM python llama.cpp\convert-hf-to-gguf.py --outtype f32 --model-name %2-f32 --outfile %1\converted\%2-f32.gguf %2

echo Conversion complete. Model saved to %1\converted\%2-%3.gguf
