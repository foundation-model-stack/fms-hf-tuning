# Standard
import re


def apply_custom_formatting_template(
    dataset, template, formatted_dataset_field, eos_token=""
):
    """Function to format datasets with Alpaca style / other templates.
    Args:
        dataset: the HF Dataset element loaded from a JSON or DatasetDict object.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}
        formatted_dataset_field: Dataset_text_field
        eos_token: string EOS token to be appended while formatting data to a single sequence.
            Defaults to empty
    Returns:
        Formatted HF Dataset
    """

    template += eos_token

    if not formatted_dataset_field:
        raise ValueError(
            "Unable to apply custom formatting because the formatted_dataset_field was not provided"
        )

    def formatter(element):
        def replace_text(match_obj):
            captured_groups = match_obj.groups()
            if len(captured_groups) != 1:
                raise ValueError(
                    "Unexpectedly captured multiple groups in template formatting"
                )

            index_object = captured_groups[0]
            if index_object not in element:
                raise KeyError("Requested template string is not a valid key in dict")

            return element[index_object]

        return {
            formatted_dataset_field: re.sub(
                r"{{([\s0-9a-zA-Z_\-\.]+)}}", replace_text, template
            )
        }

    return dataset.map(formatter)
