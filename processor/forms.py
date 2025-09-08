from django import forms

class PDFUploadForm(forms.Form):
    pdf = forms.FileField(label="PDF (plantas vetorizadas)")
    min_lotes_in_poly = forms.IntegerField(initial=6, min_value=0, required=False)
    snap_tol = forms.FloatField(initial=6.0, min_value=0, required=False)
    min_cycle_area = forms.FloatField(initial=300.0, min_value=0, required=False)
    max_imovel = forms.FloatField(initial=-1, required=False)  # -1 => auto
    max_res = forms.FloatField(initial=-1, required=False)
    debug_regions = forms.BooleanField(initial=False, required=False)