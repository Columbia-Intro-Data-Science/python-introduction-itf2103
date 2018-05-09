import re

from .allrecipes import AllRecipes

SCRAPERS = {
    AllRecipes.host(): AllRecipes,
}


def url_path_to_dict(path):
    pattern = (r'^'
               r'((?P<schema>.+?)://)?'
               r'((?P<user>.+?)(:(?P<password>.*?))?@)?'
               r'(?P<host>.*?)'
               r'(:(?P<port>\d+?))?'
               r'(?P<path>/.*?)?'
               r'(?P<query>[?].*?)?'
               r'$'
               )
    regex = re.compile(pattern)
    matches = regex.match(path)
    url_dict = matches.groupdict() if matches is not None else None

    return url_dict


class WebsiteNotImplementedError(NotImplementedError):
    '''Error for when the website is not supported by this library.'''


def scrape_me(url_path):
    host_name = url_path_to_dict(url_path.replace('://www.', '://'))['host']
    try:
        scraper = SCRAPERS[host_name]
    except KeyError:
        raise WebsiteNotImplementedError(
            "Website ({}) is not supported".format(host_name))

    return scraper(url_path)

__all__ = ['scrape_me']
