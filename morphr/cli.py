from morphr.configuration import Configuration


def main():
    configuration = Configuration.load('configuration.json')
    configuration.run()


if __name__ == '__main__':
    main()
