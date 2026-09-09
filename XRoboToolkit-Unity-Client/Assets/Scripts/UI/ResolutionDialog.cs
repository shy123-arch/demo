using System;
using TMPro;
using UnityEngine;

public class ResolutionDialog : MonoBehaviour
{
    private const string DefaultPrefsPrefix = "Resolution";

    public TextMeshProUGUI titleTmp;
    public TMP_InputField widthInputField;

    public TMP_InputField heightInputField;

    public int ResolutionWidth
    {
        get { return PlayerPrefs.GetInt("ResolutionWidth", 1024); }
        set { PlayerPrefs.SetInt("ResolutionWidth", value); }
    }

    public int ResolutionHeight
    {
        get { return PlayerPrefs.GetInt("ResolutionHeight", 768 / 2); }
        set { PlayerPrefs.SetInt("ResolutionHeight", value); }
    }

    private Action _cancel;
    private Action<int, int> _confirm;
    private string _prefsPrefix = DefaultPrefsPrefix;

    public void Show(string title, Action<int, int> call, Action cancel)
    {
        Show(title, ResolutionWidth, ResolutionHeight, DefaultPrefsPrefix, call, cancel);
    }

    public void Show(string title, int defaultWidth, int defaultHeight, string prefsPrefix, Action<int, int> call,
        Action cancel)
    {
        gameObject.SetActive(true);
        _prefsPrefix = string.IsNullOrEmpty(prefsPrefix) ? DefaultPrefsPrefix : prefsPrefix;
        titleTmp.text = title;
        widthInputField.text = PlayerPrefs.GetInt(GetWidthKey(), defaultWidth).ToString();
        heightInputField.text = PlayerPrefs.GetInt(GetHeightKey(), defaultHeight).ToString();
        _confirm = call;
        _cancel = cancel;
    }

    public void OnConfirm()
    {
        if (!int.TryParse(widthInputField.text, out int width) || width <= 0 ||
            !int.TryParse(heightInputField.text, out int height) || height <= 0)
        {
            Toast.Show("Invalid size");
            return;
        }

        gameObject.SetActive(false);
        PlayerPrefs.SetInt(GetWidthKey(), width);
        PlayerPrefs.SetInt(GetHeightKey(), height);
        PlayerPrefs.Save();
        if (_confirm != null)
        {
            _confirm.Invoke(width, height);
        }
    }

    public void OnCancel()
    {
        gameObject.SetActive(false);
        if (_cancel != null)
        {
            _cancel.Invoke();
        }
    }

    private string GetWidthKey()
    {
        return $"{_prefsPrefix}Width";
    }

    private string GetHeightKey()
    {
        return $"{_prefsPrefix}Height";
    }
}
