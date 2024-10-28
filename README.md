پروژه بازیابی تصویر بر اساس ویژگی رنگ

این پروژه یک سیستم بازیابی تصویر بر اساس محتوا است که با استفاده از هیستوگرام‌های رنگ در فضاهای RGB و HSV به شناسایی و بازیابی تصاویر مشابه از یک دیتاست می‌پردازد.
اجرا کردن کد

برای اجرای کد این پروژه، مراحل زیر را دنبال کنید:

نصب پیش‌نیازها: اطمینان حاصل کنید که تمام کتابخانه‌های مورد نیاز، از جمله NumPy، OpenCV و pandas، نصب شده‌اند. می‌توانید از دستور زیر برای نصب آن‌ها استفاده کنید:
bash

pip install numpy opencv-python pandas

اجرا کردن نوت‌بوک: فایل نوت‌بوک Jupyter به نام ComputerVision_Mostafavi_Exercise1_403-404.ipynb را در محیط Jupyter Notebook یا JupyterLab باز کنید.

اجرای سلول‌ها: پس از باز کردن فایل، به ترتیب سلول‌ها را اجرا کنید. این کد به محاسبه هیستوگرام‌های رنگ، مقایسه آن‌ها و بازیابی تصاویر مشابه می‌پردازد.

خروجی کد

پس از اجرای کد، خروجی‌ها به صورت زیر ذخیره می‌شوند:

تصاویر مشابه: 20 تصویر مشابه به همراه تصویر ورودی در پوشه output ذخیره می‌شوند. نام‌گذاری این تصاویر به صورت rank_filename انجام می‌شود.

 گزارش بازیابی: گزارشی از نتایج بازیابی شامل دقت سیستم در یک فایل اکسل قرار خواهد گرفت. این فایل در پوشه output ذخیره می‌شود.

مقادیر هیستوگرام: هیستوگرام‌های محاسبه شده برای هر تصویر در فایل‌های جداگانه با فرمت CSV ذخیره خواهند شد.

ساختار فایل‌ها

ساختار فایل‌های پروژه به شرح زیر است:



/پروژه
│
├── ComputerVision_Mostafavi_Exercise1_403-404
│   ├── ComputerVision_Mostafavi_Exercise1_403-404.ipynb
│   ├── Exercise1_Report.docx
│   └── Exercise1_Report.pdf
│
└── output

فایل نوت‌بوک: ComputerVision_Mostafavi_Exercise1_403-404.ipynb شامل کدهای مربوط به سیستم بازیابی تصویر است.

گزارش: فایل‌های Exercise1_Report.docx و Exercise1_Report.pdf شامل توضیحات جامع درباره پروژه و نتایج آن می‌باشند.

پوشه خروجی: این پوشه شامل خروجی‌های کد و نتایج بازیابی است.






Color-Based Image Retrieval Project

This project implements a content-based image retrieval system that identifies and retrieves similar images from a dataset using color histograms in RGB and HSV color spaces.
Running the Code

To execute the code for this project, follow these steps:

Install Prerequisites: Ensure that all necessary libraries, including NumPy, OpenCV, and pandas, are installed. You can use the following command to install them:
bash

pip install numpy opencv-python pandas

Run the Notebook: Open the Jupyter notebook file named ComputerVision_Mostafavi_Exercise1_403-404.ipynb in your Jupyter Notebook or JupyterLab environment.

Execute the Cells: After opening the file, run the cells in order. The code will compute color histograms, compare them, and retrieve similar images.

Code Output

After running the code, the outputs will be saved as follows:

Similar Images: 20 similar images along with the input image will be saved in the output folder. These images will be named in a rank_filename format.

Retrieval Report: A report summarizing the retrieval results, including the accuracy of the system, will be saved in an Excel file located in the output folder.

Histogram Values: The computed histograms for each image will be saved in separate CSV files.

File Structure

The file structure of the project is as follows:



/Project
│
├── ComputerVision_Mostafavi_Exercise1_403-404
│   ├── ComputerVision_Mostafavi_Exercise1_403-404.ipynb
│   ├── Exercise1_Report.docx
│   └── Exercise1_Report.pdf
│
└── output

Notebook File: ComputerVision_Mostafavi_Exercise1_403-404.ipynb contains the code for the image retrieval system.

Report: The files Exercise1_Report.docx and Exercise1_Report.pdf provide comprehensive explanations about the project and its results.

Output Folder: This folder contains the outputs of the code and retrieval results.
